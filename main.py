"""
SigGNN Main Integration Pipeline.
Links DataLoader, Features, GraphBuilder, Trainer, and Chaos Engine.
Supports M5 dataset and auto-detects GPU.

UPDATED: 
- Uses SigGNN from models.siggnn (the ONLY canonical definition)
- Multi-window training for data augmentation (key M5 winner technique)
- WRMSSE evaluation during and after training
- Non-overlapping train/val split
"""
import torch
import torch.nn as nn
import argparse
import sys
import os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

from config import get_gpu_optimized_config, get_debug_config, get_a100_optimized_config
from data.loader import M5DataLoader
from data.features import FeatureEngineer
from data.graph_builder import HierarchicalGraphBuilder
from data.wrmsse import WRMSSEEvaluator, compute_simple_metrics
from models.siggnn import SigGNN, WRMSSEAlignedLoss, BlendedLoss
from train import SigGNNTrainer
from chaos.engine import ChaosEngine
from chaos.metrics import ResilienceMetrics


def build_multi_window_training_data(
    fe, dataset, config, device, num_windows=5
):
    """
    Build multiple training windows for data augmentation.
    Each window uses a different end_day, giving the model diverse 
    temporal views of each item. This is the key M5 winner technique.
    
    Returns: list of (features, targets) tuples
    """
    lag_window = max(config.features.lags) + max(config.features.rolling_windows)
    
    # Create windows ending at different points in the training period
    # Space them 28 days apart so each has different target periods
    windows = []
    latest_train_end = config.data.train_end
    
    for i in range(num_windows):
        end_day = latest_train_end - i * 28
        if end_day - lag_window < 0:
            break
            
        data = fe.build_stream_tensors(
            dataset,
            start_day=end_day - lag_window,
            end_day=end_day,
            device=device
        )
        windows.append(data)
    
    print(f"   Created {len(windows)} training windows")
    return windows


def run_pipeline(args):
    print(f"============================================================")
    print(f"=== M5 SigGNN Pipeline: Mode={args.mode} ===")
    print(f"============================================================")
    
    if args.mode == 'debug':
        config = get_debug_config()
    elif args.mode == 'a100':
        config = get_a100_optimized_config()
    else:
        config = get_gpu_optimized_config()
        
    if args.data_dir:
        config.data.data_dir = args.data_dir

    if args.no_hawkes:
        config.chaos.use_hawkes = False

    device = config.device
    print(f"[GPU] Target Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ── 1. Data Loading ──
    loader = M5DataLoader(config.data)
    dataset = loader.prepare_dataset()
    
    # ── 2. Feature Engineering ──
    print(f"[FEAT] Engineering features...")
    fe = FeatureEngineer(config.data, config.features)
    
    lag_window = max(config.features.lags) + max(config.features.rolling_windows)
    
    # Primary training window
    train_data = fe.build_stream_tensors(
        dataset, 
        start_day=config.data.train_end - lag_window,
        end_day=config.data.train_end,
        device=device
    )
    print(f"   Train features: [{config.data.train_end - lag_window}, {config.data.train_end}]")
    print(f"   Train targets:  [{config.data.train_end}, {config.data.train_end + 28}]")
    
    # Build additional training windows for multi-window training
    num_extra_windows = 4 if args.mode != 'debug' else 1
    extra_windows = []
    for i in range(1, num_extra_windows + 1):
        end_day = config.data.train_end - i * 28
        if end_day - lag_window < 0:
            break
        w = fe.build_stream_tensors(
            dataset,
            start_day=end_day - lag_window,
            end_day=end_day,
            device=device
        )
        extra_windows.append(w)
    print(f"   Extra training windows: {len(extra_windows)}")
    
    # Validation window
    val_data = fe.build_stream_tensors(
        dataset, 
        start_day=config.data.val_start - lag_window,
        end_day=config.data.val_start,
        device=device
    )
    print(f"   Val features:   [{config.data.val_start - lag_window}, {config.data.val_start}]")
    print(f"   Val targets:    [{config.data.val_start}, {config.data.val_start + 28}]")
    
    # Compute historical means for reconciliation clipping
    train_sales = dataset['sales_matrix'][:, :config.data.train_end]
    hist_mean = torch.tensor(train_sales.mean(axis=1), dtype=torch.float32).to(device)
    train_data['historical_mean'] = hist_mean
    val_data['historical_mean'] = hist_mean
    for w in extra_windows:
        w['historical_mean'] = hist_mean
    
    # Compute DOW baselines for residual learning
    print(f"[FEAT] Computing DOW baselines...")
    train_baseline = fe.compute_dow_baseline(dataset['sales_matrix'], config.data.train_end, horizon=28, num_weeks=4)
    val_baseline = fe.compute_dow_baseline(dataset['sales_matrix'], config.data.val_start, horizon=28, num_weeks=4)
    train_data['baseline'] = torch.tensor(train_baseline, dtype=torch.float32).to(device)
    val_data['baseline'] = torch.tensor(val_baseline, dtype=torch.float32).to(device)
    for i, w in enumerate(extra_windows):
        wd_end = config.data.train_end - (i + 1) * 28
        w_baseline = fe.compute_dow_baseline(dataset['sales_matrix'], wd_end, horizon=28, num_weeks=4)
        w['baseline'] = torch.tensor(w_baseline, dtype=torch.float32).to(device)
    
    val_data['dept_ids'] = val_data['category_ids'].get('dept_id')
    train_data['dept_ids'] = train_data['category_ids'].get('dept_id')
    for w in extra_windows:
        w['dept_ids'] = w['category_ids'].get('dept_id')
    # ── 3. Graph Building ──
    gb = HierarchicalGraphBuilder()
    graph_data = gb.build_graph(
        sales_matrix=dataset['sales_matrix'],
        metadata=dataset['metadata'],
        train_end=config.data.train_end,
        device=device
    )
    
    train_data['edge_index'] = graph_data['edge_index']
    train_data['edge_type'] = graph_data['edge_type']
    val_data['edge_index'] = graph_data['edge_index']
    val_data['edge_type'] = graph_data['edge_type']
    for w in extra_windows:
        w['edge_index'] = graph_data['edge_index']
        w['edge_type'] = graph_data['edge_type']
    
    # ── 4. WRMSSE Evaluator Setup ──
    print(f"[EVAL] Setting up WRMSSE evaluator...")
    train_prices = dataset['price_matrix'][:, :config.data.train_end]
    wrmsse_evaluator = WRMSSEEvaluator(
        train_sales=train_sales,
        train_prices=train_prices,
        metadata=dataset['metadata'],
        horizon=config.data.horizon
    )
    print(f"   Scales range: [{wrmsse_evaluator.scales.min():.4f}, {wrmsse_evaluator.scales.max():.4f}]")
    print(f"   Weights range: [{wrmsse_evaluator.weights.min():.6f}, {wrmsse_evaluator.weights.max():.6f}]")
    
    # ── 5. Model Initialization ──
    print(f"[MODEL] Initializing SigGNN model...")
    
    vocab_sizes = val_data['category_vocab_sizes']
    num_features = val_data['num_features']
    
    model = SigGNN(
        input_channels=num_features,
        vocab_sizes=vocab_sizes,
        sig_windows=config.model.signature.windows,
        sig_depth=config.model.signature.depth,
        use_lead_lag=config.model.signature.use_lead_lag,
        gat_hidden=config.model.gat.hidden_dim,
        gat_heads=config.model.gat.num_heads,
        gat_layers=config.model.gat.num_layers,
        gat_edge_types=config.model.gat.edge_types,
        predictor_hidden=config.model.predictor_hidden,
        predictor_layers=config.model.predictor_layers,
        horizon=config.model.horizon,
        dropout=config.model.gat.dropout,
        num_dept_groups=vocab_sizes.get('dept_id_vocab_size', 7),
        residual_mode=True,  # Use DOW baseline + residual learning
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,}")
    print(f"   Signature output dim: {model.sig_encoder.get_output_dim()}")
    print(f"   GAT hidden dim: {config.model.gat.hidden_dim}")
    
    # ── 6. Setup trainer and loss ──
    config.train.loss_fn = 'blended'  # Use BlendedLoss for better WRMSSE optimization
    # Clear stale checkpoints
    ckpt_path = os.path.join(config.train.checkpoint_dir, 'best_model.pt')
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"   Cleared stale checkpoint: {ckpt_path}")
    trainer = SigGNNTrainer(model, config.train, device)
    
    if isinstance(trainer.criterion, BlendedLoss):
        weights_t = torch.tensor(wrmsse_evaluator.weights, dtype=torch.float32).to(device)
        scales_t = torch.tensor(wrmsse_evaluator.scales, dtype=torch.float32).to(device)
        trainer.criterion.set_weights(weights_t, scales_t)
        print(f"   BlendedLoss WRMSSE weights set")
    elif isinstance(trainer.criterion, WRMSSEAlignedLoss):
        weights_t = torch.tensor(wrmsse_evaluator.weights, dtype=torch.float32).to(device)
        scales_t = torch.tensor(wrmsse_evaluator.scales, dtype=torch.float32).to(device)
        trainer.criterion.set_weights(weights_t, scales_t)
        print(f"   WRMSSE-aligned loss weights set")
    
    # ── 7. Resume from checkpoint ──
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
    
    # ── 8. Multi-window Training ──
    if not args.eval_only:
        # Pass extra windows for multi-window training
        trainer.train(
            train_data, val_data, 
            wrmsse_evaluator=wrmsse_evaluator,
            extra_train_windows=extra_windows,
        )
    
    # ── 9. Final WRMSSE Evaluation ──
    print(f"\n[FINAL] Final Evaluation...")
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=config.train.use_amp):
            val_kwargs = {
                'category_ids': val_data.get('category_ids'),
                'dept_ids': val_data.get('dept_ids'),
                'historical_mean': val_data.get('historical_mean'),
                'baseline': val_data.get('baseline'),
            }
            final_preds = model(
                val_data['node_features'],
                val_data['edge_index'],
                val_data['edge_type'],
                **val_kwargs,
            )
    
    final_preds_np = final_preds.float().cpu().numpy()
    val_actuals_np = val_data['targets'].cpu().numpy()
    
    # Basic metrics
    basic_metrics = compute_simple_metrics(final_preds_np, val_actuals_np)
    print(f"\n   Basic Metrics:")
    for name, val in basic_metrics.items():
        print(f"      {name}: {val:.4f}")
    
    # WRMSSE
    final_wrmsse = wrmsse_evaluator.compute_wrmsse(final_preds_np, val_actuals_np)
    print(f"\n   >>> WRMSSE: {final_wrmsse:.4f} <<<")
    
    # Hierarchical breakdown
    hier_scores = wrmsse_evaluator.compute_hierarchical_wrmsse(final_preds_np, val_actuals_np)
    print(f"\n   Hierarchical Breakdown:")
    for level, score in hier_scores.items():
        print(f"      {level:20s}: {score:.4f}")
    
    # Save results
    with open('training_results.txt', 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("   SigGNN M5 Training Results\n")
        f.write("==================================================\n\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Loss function: {config.train.loss_fn}\n\n")
        f.write(f"WRMSSE: {final_wrmsse:.4f}\n\n")
        for name, val in basic_metrics.items():
            f.write(f"{name}: {val:.4f}\n")
        f.write(f"\nHierarchical WRMSSE:\n")
        for level, score in hier_scores.items():
            f.write(f"  {level:20s}: {score:.4f}\n")
        if trainer.history['train_loss']:
            f.write(f"\nTraining History:\n")
            for i, (tl, vl) in enumerate(zip(trainer.history['train_loss'], trainer.history['val_loss'])):
                f.write(f"  Epoch {i+1}: train={tl:.4f}, val={vl:.4f}\n")
    print(f"\n[DONE] Results saved to training_results.txt")
        
    # ── 10. Chaos Engineering Evaluation ──
    if not args.eval_only:
        print(f"\n[CHAOS] Running Chaos Engineering Suite...")
        chaos_engine = ChaosEngine(
            num_trials=config.chaos.num_chaos_trials,
            use_hawkes=config.chaos.use_hawkes,
            hawkes_mu_values=config.chaos.hawkes_mu_values,
            hawkes_alpha_values=config.chaos.hawkes_alpha_values,
            hawkes_beta_values=config.chaos.hawkes_beta_values,
            traces_dir=config.chaos.traces_dir
        )
        
        chaos_results = chaos_engine.run_all(
            model=model,
            node_features=val_data['node_features'],
            edge_index=val_data['edge_index'],
            edge_type=val_data['edge_type'],
            targets=val_data['targets'],
            loss_fn=trainer.criterion,
            category_ids=val_data['category_ids'],
            dept_ids=val_data.get('dept_ids'),
            historical_mean=val_data.get('historical_mean'),
        )
        
        # Results
        print("\n[CHAOS] Chaos Engineering Results:")
        summary_text = ResilienceMetrics.summary_table(chaos_results)
        print(summary_text)
        
        with open('training_results.txt', 'a', encoding='utf-8') as f:
            f.write("\n==================================================\n")
            f.write("   CHAOS ENGINEERING FINAL RESULTS\n")
            f.write("==================================================\n")
            f.write(summary_text)
            f.write("\n")
        print("\n[DONE] Chaos results appended to training_results.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigGNN Pipeline")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "debug", "a100"], 
                        help="Pipeline mode (full, debug, or a100)")
    parser.add_argument("--data-dir", type=str, default="", 
                        help="Path to M5 dataset directory")
    parser.add_argument("--no-hawkes", action="store_true", 
                        help="Disable Hawkes process (Bernoulli fallback)")
    parser.add_argument("--eval-only", action="store_true", 
                        help="Skip training, just evaluate chaos")
    parser.add_argument("--resume", type=str, default="", 
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()
    
    try:
        import signatory
    except ImportError:
        print("[INFO] Signatory not found. Truncated depth-2 manual signature will be used.")
        
    run_pipeline(args)
