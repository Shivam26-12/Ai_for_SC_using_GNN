"""
DIAGNOSTIC: Trace the entire SigGNN pipeline to find where the GNN signal dies.
This script loads data, builds the model, runs one forward pass, and prints
statistics at every layer to pinpoint the exact failure point.
"""
import torch
import torch.nn as nn
import numpy as np
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
from config import get_gpu_optimized_config, get_a100_optimized_config
from data.loader import M5DataLoader
from data.features import FeatureEngineer
from data.graph_builder import HierarchicalGraphBuilder
from data.wrmsse import WRMSSEEvaluator
from models.siggnn import SigGNN, BlendedLoss, WRMSSEAlignedLoss

parser = argparse.ArgumentParser()
parser.add_argument('--a100', action='store_true', help='Use A100 config')
args = parser.parse_args()

if args.a100:
    cfg = get_a100_optimized_config()
else:
    cfg = get_gpu_optimized_config()

cfg.data.stores = ['CA_1']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("DIAGNOSTIC: Tracing SigGNN Pipeline")
print("=" * 70)

# ── 1. Load Data ──
loader = M5DataLoader(cfg.data)
dataset = loader.prepare_dataset(cfg.data.stores)

fe = FeatureEngineer(cfg.data, cfg.features)
train_end = 1885
val_end = 1913

# Build features
item_features = fe.compute_item_features(dataset['sales_matrix'], train_end)
precomputed = {
    'lag_feats': fe.compute_lag_features(dataset['sales_matrix'], cfg.features.lags),
    'rolling_feats': fe.compute_rolling_features(dataset['sales_matrix'], cfg.features.rolling_windows),
    'price_feats': fe.compute_price_features(dataset['price_matrix']),
}

FEATURE_WINDOW = 140

# Training data
train_data = fe.build_stream_tensors(
    dataset, start_day=max(0, train_end - FEATURE_WINDOW),
    end_day=train_end, device=device, item_features=item_features,
    precomputed=precomputed
)

# Validation data (for WRMSSE evaluation)
test_data = fe.build_stream_tensors(
    dataset, start_day=max(0, val_end - FEATURE_WINDOW),
    end_day=val_end, device=device, item_features=item_features,
    precomputed=precomputed
)

# Baselines
train_baseline = fe.compute_dow_baseline(dataset['sales_matrix'], train_end, horizon=28, num_weeks=4)
test_baseline = fe.compute_dow_baseline(dataset['sales_matrix'], val_end, horizon=28, num_weeks=4)

print("\n" + "=" * 70)
print("CHECKPOINT 1: Feature Statistics")
print("=" * 70)
feats = train_data['node_features']
targets = train_data['targets']
print(f"Features shape: {feats.shape}")
print(f"Features mean: {feats.mean():.4f}, std: {feats.std():.4f}")
print(f"Features min: {feats.min():.4f}, max: {feats.max():.4f}")
print(f"Features has NaN: {torch.isnan(feats).any()}")
print(f"Features has Inf: {torch.isinf(feats).any()}")
print(f"\nTargets shape: {targets.shape}")
print(f"Targets mean: {targets.mean():.4f}")
print(f"Targets min: {targets.min():.4f}, max: {targets.max():.4f}")
print(f"Targets % zeros: {(targets == 0).float().mean() * 100:.1f}%")

bl_tensor = torch.tensor(train_baseline, dtype=torch.float32, device=device)
residuals = targets - bl_tensor
print(f"\nBaseline mean: {bl_tensor.mean():.4f}")
print(f"Residuals (targets - baseline) mean: {residuals.mean():.4f}")
print(f"Residuals std: {residuals.std():.4f}")
print(f"Residuals min: {residuals.min():.4f}, max: {residuals.max():.4f}")

# ── 2. Build Graph ──
gb = HierarchicalGraphBuilder()
graph = gb.build_graph(dataset['sales_matrix'], dataset['metadata'], train_end, device=device)
print(f"\nGraph edges: {graph['edge_index'].shape[1]}")
print(f"Edge types: {graph['edge_type'].unique().tolist()}")

# ── 3. Build Model ──
vocab_sizes = train_data['category_vocab_sizes']
num_features = train_data['num_features']

model = SigGNN(
    input_channels=num_features,
    vocab_sizes=vocab_sizes,
    sig_windows=cfg.model.signature.windows,
    sig_depth=2,
    use_lead_lag=True,
    gat_hidden=cfg.model.gat.hidden_dim,
    gat_heads=cfg.model.gat.num_heads,
    gat_layers=cfg.model.gat.num_layers,
    gat_edge_types=3,
    predictor_hidden=cfg.model.predictor_hidden,
    predictor_layers=cfg.model.predictor_layers,
    horizon=28,
    dropout=cfg.model.gat.dropout,
    num_dept_groups=vocab_sizes.get('dept_id_vocab_size', 7),
    residual_mode=True,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")

# ── 4. Hook-based forward pass to trace activations ──
print("\n" + "=" * 70)
print("CHECKPOINT 2: Layer-by-Layer Forward Pass")
print("=" * 70)

dept_ids = torch.tensor(
    dataset['metadata']['dept_id'].astype('category').cat.codes.values,
    dtype=torch.long, device=device
)
hist_mean = torch.tensor(
    dataset['sales_matrix'][:, :train_end].mean(axis=1),
    dtype=torch.float32, device=device
)

model.eval()
with torch.no_grad():
    x = feats
    
    # Step 1: Input norm
    h = model.input_norm(x)
    print(f"\n1. After InputNorm: mean={h.mean():.4f}, std={h.std():.4f}, range=[{h.min():.2f}, {h.max():.2f}]")
    
    # Step 2: Signature encoder
    sig = model.sig_encoder(h)
    print(f"2. Signature output: shape={sig.shape}, mean={sig.mean():.4f}, std={sig.std():.4f}, range=[{sig.min():.2f}, {sig.max():.2f}]")
    sig = torch.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Step 3: Category embeddings
    cat_feats = model.hier_embed(train_data['category_ids'])
    print(f"3. Category embeds: shape={cat_feats.shape}, mean={cat_feats.mean():.4f}, std={cat_feats.std():.4f}")
    
    # Step 4: Fusion
    fused = torch.cat([sig, cat_feats], dim=-1)
    h_fused = model.fusion(fused)
    print(f"4. After Fusion: shape={h_fused.shape}, mean={h_fused.mean():.4f}, std={h_fused.std():.4f}, range=[{h_fused.min():.2f}, {h_fused.max():.2f}]")
    
    # Step 5: GAT
    h_gat = model.gat(h_fused, graph['edge_index'], graph['edge_type'])
    print(f"5. After GAT: shape={h_gat.shape}, mean={h_gat.mean():.4f}, std={h_gat.std():.4f}, range=[{h_gat.min():.2f}, {h_gat.max():.2f}]")
    
    # How much did GAT change the representation?
    gat_diff = (h_gat - h_fused).abs()
    print(f"   GAT delta (|GAT_out - GAT_in|): mean={gat_diff.mean():.6f}, max={gat_diff.max():.6f}")
    
    # Step 6: Predictor (raw output BEFORE baseline addition)
    raw_pred = model.predictor(h_gat)
    print(f"6. Raw predictor output (residual): mean={raw_pred.mean():.4f}, std={raw_pred.std():.4f}, range=[{raw_pred.min():.2f}, {raw_pred.max():.2f}]")
    
    # Step 7: After baseline addition
    final_pred = bl_tensor + raw_pred
    final_pred = torch.relu(final_pred)
    print(f"7. Final pred (baseline + residual): mean={final_pred.mean():.4f}, std={final_pred.std():.4f}")
    
    # ── Key diagnostic: Is the raw prediction useful? ──
    print("\n" + "=" * 70)
    print("CHECKPOINT 3: Is the GNN adding value?")
    print("=" * 70)
    
    baseline_error = (bl_tensor - targets).abs().mean()
    model_error = (final_pred - targets).abs().mean()
    raw_residual_error = (raw_pred - residuals).abs().mean()
    
    print(f"Target actuals mean: {targets.mean():.4f}")
    print(f"Baseline MAE:  {baseline_error:.4f}")
    print(f"Model MAE:     {model_error:.4f}")
    print(f"Raw residual std: {raw_pred.std():.4f} vs actual residual std: {residuals.std():.4f}")
    
    if model_error < baseline_error:
        print(">>> GNN IS HELPING (model MAE < baseline MAE)")
    else:
        print(f">>> GNN IS HURTING by {model_error - baseline_error:.4f} (model MAE > baseline MAE)")
        print(f"    The raw predictor outputs residuals with std={raw_pred.std():.4f}")
        print(f"    But actual residuals have std={residuals.std():.4f}")
        if raw_pred.std() > residuals.std() * 2:
            print(f"    DIAGNOSIS: Predictor is outputting TOO LARGE residuals (noise)")
        elif raw_pred.std() < residuals.std() * 0.01:
            print(f"    DIAGNOSIS: Predictor is outputting NEAR-ZERO residuals (not learning)")
        print(f"    Residual correlation check:")
        # Check if residuals are at least correlated
        rp = raw_pred.cpu().flatten().numpy()
        ra = residuals.cpu().flatten().numpy()
        corr = np.corrcoef(rp, ra)[0, 1]
        print(f"    Pearson correlation(predicted_residual, actual_residual) = {corr:.4f}")

# ── 5. Loss diagnostic ──
print("\n" + "=" * 70)
print("CHECKPOINT 4: Loss Function Weights")
print("=" * 70)

wrmsse_eval = WRMSSEEvaluator(
    dataset['sales_matrix'][:, :val_end],
    dataset['price_matrix'][:, :val_end],
    dataset['metadata']
)

weights_t = torch.tensor(wrmsse_eval.weights, dtype=torch.float32).to(device)
scales_t = torch.tensor(wrmsse_eval.scales, dtype=torch.float32).to(device)

print(f"Scales: min={scales_t.min():.4f}, max={scales_t.max():.4f}, median={scales_t.median():.4f}")
print(f"Weights: min={weights_t.min():.6f}, max={weights_t.max():.6f}")

# Show what happens inside WRMSSEAlignedLoss
safe_scales = torch.clamp(scales_t, min=1.0)
item_weights = weights_t / (safe_scales ** 2)
median_w = torch.median(item_weights)
item_weights_capped = torch.clamp(item_weights, max=10.0 * median_w)
item_weights_norm = item_weights_capped / (item_weights_capped.mean() + 1e-8)

print(f"\nAfter fix - Item weights: min={item_weights_norm.min():.4f}, max={item_weights_norm.max():.4f}, ratio={item_weights_norm.max()/item_weights_norm.min():.1f}x")

# OLD way (before fix)
old_safe = torch.clamp(scales_t, min=1e-3)
old_w = weights_t / (old_safe ** 2)
old_norm = old_w / (old_w.mean() + 1e-8)
print(f"Old (broken) weights: min={old_norm.min():.4f}, max={old_norm.max():.4f}, ratio={old_norm.max()/old_norm.min():.0f}x")

# ── 6. Gradient diagnostic ──
print("\n" + "=" * 70)
print("CHECKPOINT 5: Gradient Flow Test")
print("=" * 70)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
optimizer.zero_grad()

preds = model(
    feats, graph['edge_index'], graph['edge_type'],
    train_data['category_ids'], dept_ids, hist_mean,
    baseline=bl_tensor,
)

# Use simple MSE for gradient test
loss = nn.MSELoss()(preds, targets)
loss.backward()

# Check gradients at key layers
grad_stats = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        g = param.grad
        grad_stats[name] = {
            'mean': g.abs().mean().item(),
            'max': g.abs().max().item(),
            'has_nan': torch.isnan(g).any().item()
        }

# Print top 10 by grad magnitude
sorted_grads = sorted(grad_stats.items(), key=lambda x: x[1]['max'], reverse=True)
for name, stats in sorted_grads[:15]:
    nan_warn = " *** NaN! ***" if stats['has_nan'] else ""
    print(f"  {name:50s} mean={stats['mean']:.6f}  max={stats['max']:.6f}{nan_warn}")

# Check if GAT layers get meaningful gradients
gat_grad_total = sum(v['mean'] for k, v in grad_stats.items() if 'gat' in k)
pred_grad_total = sum(v['mean'] for k, v in grad_stats.items() if 'predictor' in k)
sig_grad_total = sum(v['mean'] for k, v in grad_stats.items() if 'sig' in k)
print(f"\nTotal grad magnitude - GAT: {gat_grad_total:.6f}, Predictor: {pred_grad_total:.6f}, Signature: {sig_grad_total:.6f}")

if gat_grad_total < 1e-6:
    print("DIAGNOSIS: GAT gradients are DEAD - vanishing gradient problem!")
elif gat_grad_total > 100:
    print("DIAGNOSIS: GAT gradients are EXPLODING!")
else:
    print("Gradients look reasonable in magnitude.")

# ── 7. WRMSSE on baseline vs untrained model ──
print("\n" + "=" * 70)
print("CHECKPOINT 6: WRMSSE Baseline vs Untrained Model")
print("=" * 70)

test_bl = test_baseline
test_actuals = dataset['sales_matrix'][:, val_end:val_end+28]

model.eval()
with torch.no_grad():
    test_preds = model(
        test_data['node_features'], graph['edge_index'], graph['edge_type'],
        test_data['category_ids'], dept_ids, hist_mean,
        baseline=torch.tensor(test_bl, dtype=torch.float32, device=device),
    )
test_preds_np = test_preds.cpu().numpy()

bl_wrmsse = wrmsse_eval.compute_wrmsse(test_bl, test_actuals)
model_wrmsse = wrmsse_eval.compute_wrmsse(test_preds_np, test_actuals)

print(f"DOW Baseline WRMSSE: {bl_wrmsse:.4f}")
print(f"Untrained Model WRMSSE: {model_wrmsse:.4f}")
print(f"Difference: {model_wrmsse - bl_wrmsse:+.4f}")

if model_wrmsse > bl_wrmsse + 0.1:
    print("\nDIAGNOSIS: Even UNTRAINED model is significantly worse than baseline!")
    print("This means the model's random initialization is adding destructive noise.")
    print("The predictor output scale is too large relative to the baseline.")
    print(f"FIX NEEDED: Initialize predictor final layer to near-zero weights,")
    print(f"so the model starts as 'baseline + ~0 residual' = pure baseline.")
elif abs(model_wrmsse - bl_wrmsse) < 0.02:
    print("\nUntrained model ≈ baseline. Good initialization - model starts from baseline.")
else:
    print(f"\nUntrained model is slightly {'worse' if model_wrmsse > bl_wrmsse else 'better'} than baseline.")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
