import torch
import numpy as np
import pandas as pd
import time
import argparse
import sys
import os

from config import ExperimentConfig
from data.loader import M5DataLoader
from data.features import FeatureEngineer
from data.graph_builder import HierarchicalGraphBuilder
from data.wrmsse import WRMSSEEvaluator, compute_simple_metrics
from models.siggnn import SigGNN, TweedieLoss
from train import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=str, default='CA_1', help='Store ID to train on (e.g. CA_1, or "all" for all 10 stores)')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train for')
    args = parser.parse_args()

    # 1. Config Setup
    print("\n" + "="*50)
    print("   SigGNN M5 Submission Pipeline")
    print("="*50 + "\n")
    
    cfg = ExperimentConfig()
    cfg.data.data_dir = './dataset'
    
    if args.store.lower() != 'all':
        cfg.data.stores = [args.store]
        print(f"Mode: Single Store ({args.store})")
    else:
        cfg.data.stores = []
        print("Mode: Full M5 Dataset (All 10 stores)")
        
    cfg.train.max_epochs = args.epochs
    cfg.train.batch_size = 512
    cfg.model.gat.hidden_dim = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 2. Load Data
    loader = M5DataLoader(cfg.data)
    dataset = loader.prepare_dataset(cfg.data.stores)
    
    feat_eng = FeatureEngineer(cfg.data, cfg.features)
    
    # ── Temporal Splitting for M5 ──
    # M5 competition timeline:
    # d_1 to d_1913: Published training data
    # d_1914 to d_1941: Initial validation phase (published right before end)
    # d_1942 to d_1969: Evaluation phase (hidden targets)

    # To tune models properly without leakage, we use:
    # Train: d_1 to d_1885
    # Val: d_1886 to d_1913
    train_end = 1885
    val_end = 1913
    test_end = 1941
    
    print("\n   [Building Time Tensors]")
    print(f"   Train features end: d_{train_end}")
    print(f"   Val features end:   d_{val_end} (to predict d_1914 - d_1941)")
    print(f"   Eval features end:  d_{test_end} (to predict d_1942 - d_1969)")

    # Features for training
    train_data = feat_eng.build_stream_tensors(dataset, start_day=max(0, train_end - 90), end_day=train_end, device=device)
    
    # Features for internal validation (predicting d_1886 - d_1913)
    val_data = feat_eng.build_stream_tensors(dataset, start_day=max(0, val_end - 28 - 90), end_day=val_end - 28, device=device)
    
    # Features for final validation predictions (predicting d_1914 - d_1941)
    test_data = feat_eng.build_stream_tensors(dataset, start_day=max(0, val_end - 90), end_day=val_end, device=device)
    
    # Features for final evaluation predictions (predicting d_1942 - d_1969)
    eval_data = feat_eng.build_stream_tensors(dataset, start_day=max(0, test_end - 90), end_day=test_end, device=device)
    
    # 3. Build Graph
    graph_builder = HierarchicalGraphBuilder()
    graph = graph_builder.build_graph(dataset['sales_matrix'], dataset['metadata'], train_end, device=device)
    
    vocab_sizes = train_data['category_vocab_sizes']
    num_features = train_data['num_features']
    dept_ids = torch.tensor(dataset['metadata']['dept_id'].astype('category').cat.codes.values, dtype=torch.long, device=device)
    hist_mean = torch.tensor(dataset['sales_matrix'][:, :train_end].mean(axis=1), dtype=torch.float32, device=device)
    
    # 4. Model Definition
    model = SigGNN(
        input_channels=num_features,
        vocab_sizes=vocab_sizes,
        sig_windows=[7, 28, 90],
        sig_depth=3,
        use_lead_lag=True,
        gat_hidden=cfg.model.gat.hidden_dim,
        gat_heads=1,  # Faster for single GPU
        gat_layers=cfg.model.gat.num_layers,
        predictor_hidden=cfg.model.predictor_hidden,
        horizon=cfg.data.horizon,
    ).to(device)
    
    loss_fn = TweedieLoss(p=1.5)
    
    # 5. Training
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        device=device,
        lr=5e-4,
        max_epochs=cfg.train.max_epochs,
        use_amp=torch.cuda.is_available()
    )
    
    print("\n   [Starting Training]")
    # We use train_data for training, and val_data for early stopping tracking
    trainer.train(
        train_features=train_data['node_features'],
        train_edge_index=graph['edge_index'],
        train_edge_type=graph['edge_type'],
        train_targets=train_data['targets'],
        train_category_ids=train_data['category_ids'],
        train_dept_ids=dept_ids,
        train_historical_mean=hist_mean,
        val_features=val_data['node_features'],
        val_edge_index=graph['edge_index'], 
        val_edge_type=graph['edge_type'],
        val_targets=val_data['targets'],
        val_category_ids=val_data['category_ids'],
        val_dept_ids=dept_ids,
        val_historical_mean=hist_mean,
    )
    
    # 6. Evaluation & Submission
    print("\n==================================================")
    print("   📊 FINAL EVALUATION & SUBMISSION GENERATION")
    print("==================================================")
    model.eval()
    
    with torch.no_grad():
        # Predict d_1914 - d_1941
        val_preds = model(
            test_data['node_features'], graph['edge_index'], graph['edge_type'],
            test_data['category_ids'], dept_ids, hist_mean
        )
        # Predict d_1942 - d_1969
        eval_preds = model(
            eval_data['node_features'], graph['edge_index'], graph['edge_type'],
            eval_data['category_ids'], dept_ids, hist_mean
        )
        
    val_preds_np = val_preds.cpu().numpy()
    eval_preds_np = eval_preds.cpu().numpy()
    
    # Calculate WRMSSE on the d_1914-1941 period (for which we know the actuals!)
    actuals_1914_1941 = dataset['sales_matrix'][:, val_end:test_end]
    
    wrmsse_eval = WRMSSEEvaluator(
        train_sales=dataset['sales_matrix'][:, :val_end],
        train_prices=dataset['price_matrix'][:, :val_end],
        metadata=dataset['metadata']
    )
    
    if actuals_1914_1941.shape[1] == 28:
        wrmsse = wrmsse_eval.compute_wrmsse(val_preds_np, actuals_1914_1941)
        hier_wrmsse = wrmsse_eval.compute_hierarchical_wrmsse(val_preds_np, actuals_1914_1941)
        
        print(f"\n   {'Metric':<30} {'Score':>10}")
        print(f"   {'-'*41}")
        print(f"   {'WRMSSE (Bottom Level)':<30} {wrmsse:>10.4f}")
        print(f"   {'WRMSSE (Hierarchical - ALL)':<30} {hier_wrmsse['overall_wrmsse']:>10.4f}")
        print(f"   {'-'*41}")
        
        print("\n   [M5 Leaderboard Comparison (Approximate)]")
        print("   1st Place: 0.5015 | 2nd Place: ~0.52 | Baseline: ~0.65")
    else:
        print("   Actuals for validation block not full. Skipping WRMSSE calc.")
    
    # Format submission
    metadata_df = dataset['metadata']
    
    # The 'id' in data has '_evaluation' suffix by default in the evaluation file
    # We need to construct the matching validation IDs
    val_ids = metadata_df['id'].str.replace('_evaluation', '_validation')
    eval_ids = metadata_df['id']
    
    val_df = pd.DataFrame(val_preds_np, columns=[f'F{i}' for i in range(1, 29)])
    val_df.insert(0, 'id', val_ids)
    
    eval_df = pd.DataFrame(eval_preds_np, columns=[f'F{i}' for i in range(1, 29)])
    eval_df.insert(0, 'id', eval_ids)
    
    sub = pd.concat([val_df, eval_df], axis=0)
    sub.to_csv('submission.csv', index=False)
    
    print(f"\n✅ GENERATED submission.csv with {len(sub)} rows")
    print("   You can now submit this file to Kaggle!\n")

if __name__ == '__main__':
    main()
