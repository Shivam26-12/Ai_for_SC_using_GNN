"""
Comprehensive verification that NaN WRMSSE is fully eliminated.
Tests every component in the pipeline with adversarial edge cases.
"""
import torch
import numpy as np
import pandas as pd
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

print("=" * 60)
print("  NaN WRMSSE Verification Suite")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# TEST 1: WRMSSE Evaluator with adversarial data
# ═══════════════════════════════════════════════════════════════
print("\n[TEST 1] WRMSSE Evaluator — Edge Cases")
from data.wrmsse import WRMSSEEvaluator, compute_simple_metrics

# Case A: Normal data
train_sales = np.random.rand(100, 200).astype(np.float32) * 5
train_prices = np.random.rand(100, 200).astype(np.float32) * 10
meta = pd.DataFrame({
    'item_id': [f'item_{i}' for i in range(100)],
    'store_id': ['CA_1'] * 100,
    'dept_id': [f'dept_{i%7}' for i in range(100)],
    'cat_id': [f'cat_{i%3}' for i in range(100)],
    'state_id': ['CA'] * 100,
})

ev = WRMSSEEvaluator(train_sales, train_prices, meta)
preds = np.random.rand(100, 28).astype(np.float32) * 3
actuals = np.random.rand(100, 28).astype(np.float32) * 3
w = ev.compute_wrmsse(preds, actuals)
check("Normal data WRMSSE is finite", np.isfinite(w), f"got {w}")

# Case B: Zero sales (all items have zero history)
zero_sales = np.zeros((50, 200), dtype=np.float32)
zero_prices = np.zeros((50, 200), dtype=np.float32)
meta2 = pd.DataFrame({
    'item_id': [f'item_{i}' for i in range(50)],
    'store_id': ['CA_1'] * 50,
    'dept_id': ['dept_0'] * 50,
    'cat_id': ['cat_0'] * 50,
    'state_id': ['CA'] * 50,
})
ev2 = WRMSSEEvaluator(zero_sales, zero_prices, meta2)
w2 = ev2.compute_wrmsse(np.ones((50, 28)), np.zeros((50, 28)))
check("Zero-sales WRMSSE is finite", np.isfinite(w2), f"got {w2}")

# Case C: NaN-infested data
nan_sales = np.full((30, 100), np.nan, dtype=np.float32)
nan_prices = np.full((30, 100), np.nan, dtype=np.float32)
meta3 = pd.DataFrame({
    'item_id': [f'item_{i}' for i in range(30)],
    'store_id': ['CA_1'] * 30,
    'dept_id': ['dept_0'] * 30,
    'cat_id': ['cat_0'] * 30,
    'state_id': ['CA'] * 30,
})
ev3 = WRMSSEEvaluator(nan_sales, nan_prices, meta3)
w3 = ev3.compute_wrmsse(np.ones((30, 28)), np.zeros((30, 28)))
check("NaN-input WRMSSE is finite", np.isfinite(w3), f"got {w3}")

# Case D: One item with huge predictions
big_preds = np.zeros((100, 28), dtype=np.float32)
big_preds[0] = 1e6  # One item predicts millions
w4 = ev.compute_wrmsse(big_preds, actuals)
check("Huge prediction WRMSSE is finite", np.isfinite(w4), f"got {w4}")

# Case E: Hierarchical WRMSSE
hier = ev.compute_hierarchical_wrmsse(preds, actuals)
all_finite = all(np.isfinite(v) for v in hier.values())
check("Hierarchical WRMSSE all finite", all_finite, 
      f"NaN keys: {[k for k,v in hier.items() if not np.isfinite(v)]}")

# Case F: compute_simple_metrics with NaN inputs
sm = compute_simple_metrics(
    np.array([[np.nan, 1.0, np.inf]]),
    np.array([[0.0, 1.0, 2.0]])
)
all_sm_finite = all(np.isfinite(v) for v in sm.values())
check("Simple metrics with NaN input finite", all_sm_finite, f"got {sm}")

# ═══════════════════════════════════════════════════════════════
# TEST 2: Model forward pass — NaN-free output
# ═══════════════════════════════════════════════════════════════
print("\n[TEST 2] Model Forward Pass — NaN Detection")
from models.siggnn import SigGNN, BlendedLoss, WRMSSEAlignedLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

vocab = {
    'store_id_vocab_size': 1,
    'dept_id_vocab_size': 7,
    'cat_id_vocab_size': 3,
    'state_id_vocab_size': 1,
    'item_id_vocab_size': 50,
}

model = SigGNN(
    input_channels=26,
    vocab_sizes=vocab,
    sig_windows=[7, 14, 28],
    sig_depth=2,
    use_lead_lag=True,
    gat_hidden=64,
    gat_heads=4,
    gat_layers=2,
    gat_edge_types=3,
    predictor_hidden=128,
    predictor_layers=2,
    horizon=28,
    dropout=0.1,
    num_dept_groups=7,
    residual_mode=True,
).to(device)

N = 50
T = 140

# Normal input
feats = torch.randn(N, T, 26, device=device)
edge_index = torch.randint(0, N, (2, 200), device=device)
edge_type = torch.randint(0, 3, (200,), device=device)
cat_ids = {
    'store_id': torch.zeros(N, dtype=torch.long, device=device),
    'dept_id': torch.randint(0, 7, (N,), device=device),
    'cat_id': torch.randint(0, 3, (N,), device=device),
    'state_id': torch.zeros(N, dtype=torch.long, device=device),
    'item_id': torch.arange(N, device=device),
}
baseline = torch.rand(N, 28, device=device) * 3

with torch.no_grad():
    out = model(feats, edge_index, edge_type, cat_ids, baseline=baseline)
check("Normal input → finite output", torch.isfinite(out).all().item(),
      f"NaN count: {torch.isnan(out).sum().item()}")
check("Output shape correct", out.shape == (N, 28), f"got {out.shape}")
check("Output non-negative (residual mode)", (out >= 0).all().item(),
      f"min={out.min().item():.4f}")

# Adversarial: NaN input features
nan_feats = torch.full((N, T, 26), float('nan'), device=device)
with torch.no_grad():
    out_nan = model(nan_feats, edge_index, edge_type, cat_ids, baseline=baseline)
check("NaN input → finite output", torch.isfinite(out_nan).all().item(),
      f"NaN count: {torch.isnan(out_nan).sum().item()}")

# Adversarial: Huge input features
huge_feats = torch.full((N, T, 26), 1e4, device=device)
with torch.no_grad():
    out_huge = model(huge_feats, edge_index, edge_type, cat_ids, baseline=baseline)
check("Huge input → finite output", torch.isfinite(out_huge).all().item(),
      f"NaN count: {torch.isnan(out_huge).sum().item()}")

# ═══════════════════════════════════════════════════════════════
# TEST 3: Loss Functions — NaN-free
# ═══════════════════════════════════════════════════════════════
print("\n[TEST 3] Loss Functions — NaN Safety")

# BlendedLoss
blended = BlendedLoss()
weights_t = torch.rand(N, device=device)
weights_t = weights_t / weights_t.sum()
scales_t = torch.rand(N, device=device) * 5 + 0.1
blended.set_weights(weights_t, scales_t)

preds_t = torch.randn(N, 28, device=device, requires_grad=True)
targets_t = torch.randn(N, 28, device=device)
loss = blended(preds_t, targets_t)
check("BlendedLoss finite", torch.isfinite(loss).item(), f"got {loss.item()}")
check("BlendedLoss has grad", loss.requires_grad, "no gradient!")

# BlendedLoss with near-zero scales
tiny_scales = torch.full((N,), 1e-8, device=device)
blended2 = BlendedLoss()
blended2.set_weights(weights_t, tiny_scales)
loss2 = blended2(preds_t, targets_t)
check("BlendedLoss tiny scales → finite", torch.isfinite(loss2).item(), f"got {loss2.item()}")

# WRMSSEAlignedLoss with NaN predictions
wloss = WRMSSEAlignedLoss()
wloss.set_weights(weights_t, scales_t)
nan_preds = torch.full((N, 28), float('nan'), device=device)
loss3 = wloss(nan_preds, targets_t)
check("WRMSSEAlignedLoss NaN preds → finite", torch.isfinite(loss3).item(), f"got {loss3}")

# ═══════════════════════════════════════════════════════════════
# TEST 4: Training Loop — One epoch without crash
# ═══════════════════════════════════════════════════════════════
print("\n[TEST 4] Training Loop — Single Epoch")
from config import TrainConfig
from train import SigGNNTrainer

cfg = TrainConfig()
cfg.max_epochs = 1
cfg.use_amp = torch.cuda.is_available()
cfg.amp_dtype = 'bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'float16'
cfg.loss_fn = 'blended'
cfg.checkpoint_dir = ''  # No checkpointing for test

trainer = SigGNNTrainer(model, cfg, device)

# Set weights on blended loss
if isinstance(trainer.criterion, BlendedLoss):
    trainer.criterion.set_weights(weights_t, scales_t)

targets = torch.rand(N, 28, device=device) * 5
loss = trainer.train_epoch(
    feats, edge_index, edge_type, targets,
    category_ids=cat_ids, baseline=baseline,
)
check("Train epoch loss finite", np.isfinite(loss), f"got {loss}")

# Evaluate
val_loss, val_preds = trainer.evaluate(
    feats, edge_index, edge_type, targets,
    category_ids=cat_ids, baseline=baseline,
)
check("Eval loss finite", np.isfinite(val_loss), f"got {val_loss}")
check("Eval preds all finite", np.all(np.isfinite(val_preds)), 
      f"NaN count: {np.isnan(val_preds).sum()}")

# ═══════════════════════════════════════════════════════════════
# TEST 5: WRMSSE on model predictions (end-to-end)
# ═══════════════════════════════════════════════════════════════
print("\n[TEST 5] End-to-End: Model Preds → WRMSSE")
small_actuals = np.random.rand(N, 28).astype(np.float32) * 3
small_sales = np.random.rand(N, 200).astype(np.float32) * 5
small_prices = np.random.rand(N, 200).astype(np.float32) * 10
meta_small = pd.DataFrame({
    'item_id': [f'item_{i}' for i in range(N)],
    'store_id': ['CA_1'] * N,
    'dept_id': [f'dept_{i%7}' for i in range(N)],
    'cat_id': [f'cat_{i%3}' for i in range(N)],
    'state_id': ['CA'] * N,
})

ev_small = WRMSSEEvaluator(small_sales, small_prices, meta_small)
e2e_wrmsse = ev_small.compute_wrmsse(val_preds, small_actuals)
check("End-to-end WRMSSE finite", np.isfinite(e2e_wrmsse), f"got {e2e_wrmsse}")

e2e_hier = ev_small.compute_hierarchical_wrmsse(val_preds, small_actuals)
all_hier_finite = all(np.isfinite(v) for v in e2e_hier.values())
check("End-to-end hierarchical WRMSSE finite", all_hier_finite,
      f"NaN keys: {[k for k,v in e2e_hier.items() if not np.isfinite(v)]}")

# ═══════════════════════════════════════════════════════════════
# TEST 6: A100 Config Verification
# ═══════════════════════════════════════════════════════════════
print("\n[TEST 6] A100 Configuration")
from config import get_a100_optimized_config

a100_cfg = get_a100_optimized_config()
check("A100 uses BF16", a100_cfg.train.amp_dtype == 'bfloat16', 
      f"got {a100_cfg.train.amp_dtype}")
check("A100 sig windows no 90-day", 90 not in a100_cfg.model.signature.windows,
      f"windows: {a100_cfg.model.signature.windows}")
check("A100 hidden dim 128", a100_cfg.model.gat.hidden_dim == 128,
      f"got {a100_cfg.model.gat.hidden_dim}")
check("A100 8 heads", a100_cfg.model.gat.num_heads == 8,
      f"got {a100_cfg.model.gat.num_heads}")
check("A100 3 GAT layers", a100_cfg.model.gat.num_layers == 3,
      f"got {a100_cfg.model.gat.num_layers}")
check("A100 patience 40", a100_cfg.train.patience == 40,
      f"got {a100_cfg.train.patience}")

# ═══════════════════════════════════════════════════════════════
# TEST 7: Signature Encoder — Clamp & NaN guard
# ═══════════════════════════════════════════════════════════════
print("\n[TEST 7] Signature Encoder")
from models.signature import manual_signature_depth2, MultiScaleSignatureEncoder

# Normal path
path = torch.randn(4, 50, 16)
sig = manual_signature_depth2(path)
check("Normal signature finite", torch.isfinite(sig).all().item(),
      f"NaN: {torch.isnan(sig).sum().item()}")
check("Signature clamped ≤ 30", sig.abs().max().item() <= 30.0,
      f"max abs: {sig.abs().max().item():.2f}")

# Extreme path
big_path = torch.full((4, 50, 16), 100.0)
sig_big = manual_signature_depth2(big_path)
check("Extreme signature finite", torch.isfinite(sig_big).all().item(),
      f"NaN: {torch.isnan(sig_big).sum().item()}")
check("Extreme signature clamped ≤ 30", sig_big.abs().max().item() <= 30.0,
      f"max abs: {sig_big.abs().max().item():.2f}")

# NaN path
nan_path = torch.full((4, 50, 16), float('nan'))
sig_nan = manual_signature_depth2(nan_path)
check("NaN-path signature finite", torch.isfinite(sig_nan).all().item(),
      f"NaN: {torch.isnan(sig_nan).sum().item()}")

# ═══════════════════════════════════════════════════════════════
# TEST 8: GAT Layer — Device type detection
# ═══════════════════════════════════════════════════════════════
print("\n[TEST 8] GAT Layer")
from models.gat import SparseGATLayer, _get_device_type

cpu_tensor = torch.randn(10)
check("CPU device type = 'cpu'", _get_device_type(cpu_tensor) == 'cpu')
if torch.cuda.is_available():
    gpu_tensor = torch.randn(10, device='cuda')
    check("CUDA device type = 'cuda'", _get_device_type(gpu_tensor) == 'cuda')

# GAT forward
gat = SparseGATLayer(64, 64, num_heads=4, num_edge_types=3).to(device)
x = torch.randn(N, 64, device=device)
ei = torch.randint(0, N, (2, 100), device=device)
et = torch.randint(0, 3, (100,), device=device)
with torch.no_grad():
    out_gat = gat(x, ei, et)
check("GAT output finite", torch.isfinite(out_gat).all().item(),
      f"NaN: {torch.isnan(out_gat).sum().item()}")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("  🎉 ALL TESTS PASSED — NaN WRMSSE is ELIMINATED!")
    print("  ✅ Safe to train on A100 with: python run_m5.py --a100")
else:
    print(f"  ⚠️ {FAIL} test(s) failed — review above")
print("=" * 60)
