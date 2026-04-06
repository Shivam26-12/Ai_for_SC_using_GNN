"""
SigGNN Main Integration Pipeline.
Links DataLoader, Features, GraphBuilder, Trainer, and Chaos Engine.
Supports M5 dataset and auto-detects GPU.
"""
import torch
import torch.nn as nn
import argparse
import sys
import os

from config import get_gpu_optimized_config, get_debug_config
from data.loader import M5DataLoader
from data.features import FeatureEngineer
from data.graph_builder import HierarchicalGraphBuilder
from data.wrmsse import WRMSSEEvaluator
from models.signature import MultiScaleSignatureEncoder
from models.gat import SparseTemporalGAT
from models.reconciliation import HierarchicalReconciliation, SimpleReconciliation
from train import SigGNNTrainer
from chaos.engine import ChaosEngine
from chaos.metrics import ResilienceMetrics


class SigGNN(nn.Module):
    """Full End-to-End SigGNN Model."""
    def __init__(self, config, vocab_sizes):
        super().__init__()
        self.config = config
        
        # ── 1. Signature Encoder ──
        c_sig = config.model.signature
        self.signature = MultiScaleSignatureEncoder(
            input_channels=c_sig.input_channels,
            windows=c_sig.windows,
            depth=c_sig.depth,
            use_lead_lag=c_sig.use_lead_lag,
            use_logsig=c_sig.use_logsig,
            projection_dim=16
        )
        sig_out_dim = self.signature.get_output_dim()
        
        # ── 2. Categorical Embeddings ──
        self.embeddings = nn.ModuleDict()
        embed_dims = {
            'store_id': config.features.store_embed_dim,
            'dept_id': config.features.dept_embed_dim,
            'cat_id': config.features.cat_embed_dim,
            'state_id': config.features.state_embed_dim,
            'item_id': config.features.item_embed_dim,
        }
        total_embed_dim = 0
        for name, size in vocab_sizes.items():
            col = name.replace('_vocab_size', '')
            if col in embed_dims:
                dim = embed_dims[col]
                self.embeddings[col] = nn.Embedding(size, dim)
                total_embed_dim += dim
                
        # ── 3. Graph Attention Network ──
        c_gat = config.model.gat
        gat_in_dim = sig_out_dim + total_embed_dim
        self.gat = SparseTemporalGAT(
            in_dim=gat_in_dim,
            hidden_dim=c_gat.hidden_dim,
            out_dim=c_gat.hidden_dim,
            num_heads=c_gat.num_heads,
            num_layers=c_gat.num_layers,
            num_edge_types=c_gat.edge_types,
            dropout=c_gat.dropout,
            residual=c_gat.residual,
            layer_norm=c_gat.layer_norm
        )
        
        # ── 4. Predictor MLP ──
        self.predictor = nn.Sequential(
            nn.Linear(c_gat.hidden_dim, config.model.predictor_hidden),
            nn.GELU(),
            nn.Dropout(config.model.predictor_dropout),
            nn.Linear(config.model.predictor_hidden, config.model.horizon)
        )
        
        # ── 5. Reconciliation (Simple form for now) ──
        num_depts = vocab_sizes.get('dept_id_vocab_size', 7)
        self.reconciliation = SimpleReconciliation(num_groups=num_depts)
        
    def forward(self, features, edge_index, edge_type, category_ids, dept_ids=None, historical_mean=None):
        # 1. Signature
        sig = self.signature(features)
        
        # 2. Embeddings
        embs = []
        for name, emb_layer in self.embeddings.items():
            if name in category_ids:
                embs.append(emb_layer(category_ids[name]))
        
        if embs:
            node_emb = torch.cat([sig] + embs, dim=-1)
        else:
            node_emb = sig
            
        # 3. GAT
        h = self.gat(node_emb, edge_index, edge_type)
        
        # 4. Predict
        raw_pred = self.predictor(h)
        
        # 5. Reconcile
        if dept_ids is None and 'dept_id' in category_ids:
            dept_ids = category_ids['dept_id']
            
        final_pred = self.reconciliation(raw_pred, group_ids=dept_ids, historical_mean=historical_mean)
        return final_pred


def run_pipeline(args):
    print(f"============================================================")
    print(f"=== M5 SigGNN Pipeline: Mode={args.mode} ===")
    print(f"============================================================")
    
    if args.mode == 'debug':
        config = get_debug_config()
    else:
        config = get_gpu_optimized_config()
        
    if args.data_dir:
        config.data.data_dir = args.data_dir

    if args.no_hawkes:
        config.chaos.use_hawkes = False

    device = config.device
    print(f"🖥️ Target Device: {device}")
    
    # ── 1. Data Loading ──
    loader = M5DataLoader(config.data)
    dataset = loader.prepare_dataset()
    
    # ── 2. Feature Engineering ──
    print(f"⚙️ Engineering features...")
    fe = FeatureEngineer(config.data, config.features)
    
    # Validation window (t-28 to t)
    val_data = fe.build_stream_tensors(
        dataset, 
        start_day=config.data.val_start - config.features.lags[-1] - max(config.features.rolling_windows),
        end_day=config.data.val_start,
        device=device
    )
    
    # Training window (t-56 to t-28)
    train_data = fe.build_stream_tensors(
        dataset, 
        start_day=config.data.train_end - config.features.lags[-1] - max(config.features.rolling_windows),
        end_day=config.data.train_end,
        device=device
    )
    
    # Compute historical means for clipping
    train_sales = dataset['sales_matrix'][:, :config.data.train_end]
    hist_mean = torch.tensor(train_sales.mean(axis=1), dtype=torch.float32).to(device)
    train_data['historical_mean'] = hist_mean
    val_data['historical_mean'] = hist_mean
    
    val_data['dept_ids'] = val_data['category_ids'].get('dept_id')
    train_data['dept_ids'] = train_data['category_ids'].get('dept_id')
    
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
    
    # ── 4. Model Initialization ──
    print(f"🧠 Initializing SigGNN model...")
    model = SigGNN(config, val_data['category_vocab_sizes'])
    print(f"   Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ── 5. Training ──
    trainer = SigGNNTrainer(model, config.train, device)
    
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
    
    if not args.eval_only:
        trainer.train(train_data, val_data)
        
    # ── 6. Chaos Engineering Evaluation ──
    print(f"\n🌪️ Running Chaos Engineering Suite...")
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
    
    # ── 7. Results ──
    print("\n📊 Chaos Engineering Results:")
    print(ResilienceMetrics.summary_table(chaos_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigGNN Pipeline")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "debug"], 
                        help="Pipeline mode (full or debug)")
    parser.add_argument("--data-dir", type=str, default="", 
                        help="Path to M5 dataset directory")
    parser.add_argument("--no-hawkes", action="store_true", 
                        help="Disable Hawkes process (Bernoulli fallback)")
    parser.add_argument("--eval-only", action="store_true", 
                        help="Skip training, just evaluate chaos")
    parser.add_argument("--resume", type=str, default="", 
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()
    
    # Automatically install signatory if missing on an environment that supports it
    try:
        import signatory
    except ImportError:
        print("⚠️ Signatory not found. Truncated depth-2 manual signature will be used.")
        
    run_pipeline(args)
