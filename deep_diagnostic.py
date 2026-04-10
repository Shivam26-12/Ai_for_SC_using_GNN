import torch
import numpy as np
import sys
import os
from torch.cuda.amp import autocast

# Force absolute imports
sys.path.append(os.getcwd())

import argparse
from config import get_gpu_optimized_config, get_a100_optimized_config
from data.loader import M5DataLoader
from data.features import FeatureEngineer
from models.siggnn import SigGNN, TweedieLoss
from models.graph_builder import GraphBuilder

def run_forensic_audit(use_a100=False):
    print("🚀 --- STARTING DEEP-DIVE FORENSIC AUDIT ---")
    if use_a100:
        cfg = get_a100_optimized_config()
    else:
        cfg = get_gpu_optimized_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    print("\n1. Loading Full Dataset...")
    loader = M5DataLoader(cfg.data)
    dataset = loader.prepare_dataset()
    fe = FeatureEngineer(cfg.data, cfg.features)
    
    # 2. Build Features for one batch
    print("2. Building Features...")
    batch = fe.build_stream_tensors(dataset, 1000, 1100, device=device)
    node_features = batch['node_features']
    targets = batch['targets']
    
    # Check inputs
    if torch.isnan(node_features).any():
        print("❌ CRITICAL: NaNs found in node_features before model!")
        return

    # 3. Build Graph
    print("3. Building Graph...")
    gb = GraphBuilder(cfg.data)
    edge_index, edge_type = gb.build_graph(dataset['metadata'])
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    # 4. Initialize Model
    print("4. Initializing Model & Loss...")
    model = SigGNN(
        input_channels=batch['num_features'],
        vocab_sizes=batch['category_vocab_sizes']
    ).to(device)
    criterion = TweedieLoss(p=cfg.train.tweedie_p)

    # 5. THE AUDIT: Forward Pass Step-by-Step
    print("\n5. Running Step-by-Step Forward Pass Audit...")
    try:
        with autocast(enabled=cfg.train.use_amp):
            # A. Signature Layer
            sig = model.sig_encoder(node_features)
            print(f"   - Signature Output: NaNs? {torch.isnan(sig).any()}")
            
            # B. Embedding Layer
            cat_feat = model.hier_embed(batch['category_ids'])
            print(f"   - Category Embeds: NaNs? {torch.isnan(cat_feat).any()}")
            
            # C. GAT Layer
            h = torch.cat([sig, cat_feat], dim=-1)
            h = model.fusion(h)
            h_gat = model.gat(h, edge_index, edge_type)
            print(f"   - GAT Output: NaNs? {torch.isnan(h_gat).any()}")
            
            # D. Predictions
            preds = model.predictor(h_gat)
            print(f"   - Predictions: NaNs? {torch.isnan(preds).any()} | Max: {preds.max().item()} | Min: {preds.min().item()}")
            
            # E. Loss Calculation
            loss = criterion(preds, targets)
            print(f"   - Final Loss: {loss.item()}")

    except Exception as e:
        print(f"💥 FORWARD PASS CRASHED: {e}")

    # 6. Backward Pass Audit
    print("\n6. Running Backward Pass Audit (Checking Gradients)...")
    if not torch.isnan(loss):
        loss.backward()
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"   ❌ NaN Gradient found in: {name}")
                has_nan_grad = True
        
        if not has_nan_grad:
            print("   ✅ All gradients are clean.")
    else:
        print("   ⚠️ Skipping backward pass because loss is NaN.")

    print("\n🏁 --- AUDIT COMPLETE ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--a100', action='store_true', help='Use A100 configurations')
    args = parser.parse_args()
    run_forensic_audit(use_a100=args.a100)