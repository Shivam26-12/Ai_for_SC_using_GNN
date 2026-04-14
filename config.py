"""
SigGNN Configuration — All hyperparameters and paths in one place.
Designed for reproducibility in research experiments.

UPDATED: 
- Added amp_dtype for BF16 support on A100
- A100 config uses BF16 (no overflow risk), larger capacity
- Removed 90-day signature window (primary NaN source)
"""
import torch
import torch.nn as nn
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ═══════════════════════════════════════════════════════════════
# Detect environment
# ═══════════════════════════════════════════════════════════════
def _detect_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

IS_COLAB = os.path.exists('/content')

@dataclass
class DataConfig:
    """Data paths and loading configuration."""
    data_dir: str = '/content/drive/MyDrive/M6_data' if IS_COLAB else './dataset'
    sales_file: str = 'sales_train_evaluation.csv'
    calendar_file: str = 'calendar.csv'
    prices_file: str = 'sell_prices.csv'

    stores: List[str] = field(default_factory=lambda: ['CA_1']) 
    max_items: int = 0  # 0 = all items

    total_days: int = 1941 
    train_end: int = 1857   # Shifted back 28 days so train/val targets don't overlap
    val_start: int = 1886 
    val_end: int = 1913 
    test_start: int = 1914 
    test_end: int = 1941 
    horizon: int = 28 

    @property
    def sales_path(self):
        return os.path.join(self.data_dir, self.sales_file)

    @property
    def calendar_path(self):
        return os.path.join(self.data_dir, self.calendar_file)

    @property
    def prices_path(self):
        return os.path.join(self.data_dir, self.prices_file)


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    lags: List[int] = field(default_factory=lambda: [7, 14, 21, 28, 56, 84])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 28, 56])
    use_snap: bool = True
    use_events: bool = True
    use_price_momentum: bool = True
    price_momentum_window: int = 7

    store_embed_dim: int = 8
    dept_embed_dim: int = 8
    cat_embed_dim: int = 4
    state_embed_dim: int = 4
    item_embed_dim: int = 16


@dataclass
class SignatureConfig:
    """Path signature encoder parameters."""
    windows: List[int] = field(default_factory=lambda: [7, 14, 28])
    depth: int = 2  # Manual implementation only supports depth 2
    use_logsig: bool = False 
    use_lead_lag: bool = True
    input_channels: int = 26  # Updated to match feature engineer output

    @property
    def sig_input_channels(self):
        return self.input_channels * 2 if self.use_lead_lag else self.input_channels


@dataclass
class GATConfig:
    """Graph Attention Network parameters."""
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.2
    edge_types: int = 3 
    residual: bool = True
    layer_norm: bool = True


@dataclass
class ModelConfig:
    """Master model configuration."""
    signature: SignatureConfig = field(default_factory=SignatureConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    predictor_hidden: int = 128
    predictor_layers: int = 2
    predictor_dropout: float = 0.3
    horizon: int = 28


@dataclass
class TrainConfig:
    """Training configuration."""
    lr: float = 2e-4
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 30
    min_delta: float = 1e-4

    loss_fn: str = 'huber'
    tweedie_p: float = 1.5 

    scheduler: str = 'cosine'
    warmup_epochs: int = 5

    gradient_clip: float = 1.0
    label_smoothing: float = 0.0

    # GNN should use full-batch for correct message passing
    batch_size: int = 0           # 0 = full batch
    num_workers: int = 2 
    use_amp: bool = True
    amp_dtype: str = 'float16'    # 'float16' or 'bfloat16' (A100)

    adversarial_training: bool = False
    adversarial_ratio: float = 0.3
    adversarial_epsilon: float = 0.01

    num_ensemble: int = 1
    checkpoint_dir: str = './checkpoints'
    save_every: int = 10 
    resume_from: str = '' 

    pin_memory: bool = True
    cudnn_benchmark: bool = True


@dataclass
class ChaosConfig:
    """Chaos engineering configuration."""
    demand_shock_severity: float = 0.5
    demand_shock_window: int = 14
    supply_disruption_prob: float = 0.1
    supply_disruption_window: int = 14
    price_volatility_scale: float = 0.3
    calendar_shift_days: int = 3
    graph_corruption_ratio: float = 0.2
    num_chaos_trials: int = 5
    use_hawkes: bool = True

    hawkes_mu_values: List[float] = field(default_factory=lambda: [0.1])
    hawkes_alpha_values: List[float] = field(default_factory=lambda: [0.6])
    hawkes_beta_values: List[float] = field(default_factory=lambda: [1.0])
    traces_dir: str = './experiments/intensity_traces'


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    chaos: ChaosConfig = field(default_factory=ChaosConfig)
    device: torch.device = field(default_factory=_detect_device)
    seed: int = 42
    experiment_name: str = 'siggnn_v2'

    def __post_init__(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            if self.train.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════
# Preset configurations
# ═══════════════════════════════════════════════════════════════

def get_debug_config():
    """Minimal config for debugging."""
    cfg = ExperimentConfig()
    cfg.data.max_items = 100
    cfg.train.max_epochs = 10
    cfg.train.num_ensemble = 1
    cfg.model.gat.num_layers = 1
    cfg.chaos.use_hawkes = False
    cfg.experiment_name = 'debug'
    return cfg

def get_gpu_optimized_config():
    """Config optimized for RTX 4050 6GB VRAM."""
    cfg = ExperimentConfig()
    cfg.data.stores = ['CA_1']
    cfg.train.batch_size = 0       # Full batch for GNN correctness
    cfg.train.use_amp = True
    cfg.train.amp_dtype = 'float16'
    cfg.train.lr = 2e-4
    cfg.train.max_epochs = 200
    cfg.train.num_ensemble = 1
    cfg.model.gat.hidden_dim = 64
    cfg.model.gat.num_layers = 2
    cfg.experiment_name = 'gpu_rtx4050'
    return cfg

def get_a100_optimized_config():
    """Config aggressively optimized for A100 40GB/80GB (Lightning AI).
    
    Key differences from RTX 4050 config:
    - BF16 instead of FP16 (same exponent range as FP32 → no overflow!)
    - Larger model capacity (128 hidden, 8 heads, 3 layers)
    - Wider predictor (256 hidden, 3 layers)
    - No 90-day signature window (was primary NaN source)
    """
    cfg = ExperimentConfig()
    cfg.train.batch_size = 0       
    cfg.train.use_amp = True       
    cfg.train.amp_dtype = 'bfloat16'  # ← KEY: BF16 on A100, no overflow!
    cfg.train.lr = 3e-4            
    cfg.train.max_epochs = 200
    cfg.train.patience = 40
    cfg.train.num_ensemble = 1
    
    # 🚀 Unleash Model Capacity (safe with BF16)
    cfg.model.signature.windows = [7, 14, 28]  # NO 90-day window
    cfg.model.gat.hidden_dim = 128
    cfg.model.gat.num_layers = 3
    cfg.model.gat.num_heads = 8
    cfg.model.gat.dropout = 0.15
    cfg.model.predictor_hidden = 256
    cfg.model.predictor_layers = 3
    
    cfg.experiment_name = 'gpu_a100'
    return cfg