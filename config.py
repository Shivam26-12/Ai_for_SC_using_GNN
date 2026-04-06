"""
SigGNN Configuration — All hyperparameters and paths in one place.
Designed for reproducibility in research experiments.
Optimized for RTX 4050 6GB VRAM.
"""
import torch
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
    # ── Paths (override for local vs Colab) ──
    data_dir: str = '/content/drive/MyDrive/M5_data' if IS_COLAB else './dataset'
    sales_file: str = 'sales_train_evaluation.csv'
    calendar_file: str = 'calendar.csv'
    prices_file: str = 'sell_prices.csv'

    # ── Subset control ──
    # Phase 1: Single store for fast iteration
    # Phase 2: All stores for final results
    stores: List[str] = field(default_factory=lambda: ['CA_1'])  # Start small
    max_items: int = 0  # 0 = all items in selected stores

    # ── Time splits (day indices, 1-indexed matching d_1, d_2, ...) ──
    total_days: int = 1941  # d_1 to d_1941 in sales_train_evaluation
    train_end: int = 1885   # Last day of training
    val_start: int = 1886   # First day of validation (28-day window)
    val_end: int = 1913     # Last day of validation
    test_start: int = 1914  # First day of test (28-day window)
    test_end: int = 1941    # Last day of test
    horizon: int = 28       # Forecast horizon

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
    # ── Lag features ──
    lags: List[int] = field(default_factory=lambda: [7, 14, 21, 28, 56, 84])

    # ── Rolling window statistics ──
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 28, 56])

    # ── Calendar features ──
    use_snap: bool = True
    use_events: bool = True

    # ── Price features ──
    use_price_momentum: bool = True
    price_momentum_window: int = 7

    # ── Embedding dimensions for categorical features ──
    store_embed_dim: int = 8
    dept_embed_dim: int = 8
    cat_embed_dim: int = 4
    state_embed_dim: int = 4
    item_embed_dim: int = 16


@dataclass
class SignatureConfig:
    """Path signature encoder parameters."""
    # ── Multi-scale windows (short / medium / long) ──
    windows: List[int] = field(default_factory=lambda: [7, 28, 90])

    # ── Signature depth (truncation level) ──
    depth: int = 3

    # ── Use log-signature for compression ──
    use_logsig: bool = False  # Set True if OOM

    # ── Lead-lag augmentation (captures quadratic variation) ──
    use_lead_lag: bool = True

    # ── Input channels per timestep before signature ──
    # [demand, price, snap, sin_7, cos_7, sin_365, cos_365]
    input_channels: int = 7

    # After lead-lag, channels double
    @property
    def sig_input_channels(self):
        return self.input_channels * 2 if self.use_lead_lag else self.input_channels


@dataclass
class GATConfig:
    """Graph Attention Network parameters."""
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.2
    edge_types: int = 3  # hierarchical, correlation, cross-store
    residual: bool = True
    layer_norm: bool = True


@dataclass
class ModelConfig:
    """Master model configuration."""
    signature: SignatureConfig = field(default_factory=SignatureConfig)
    gat: GATConfig = field(default_factory=GATConfig)

    # ── Predictor MLP ──
    predictor_hidden: int = 256
    predictor_layers: int = 3
    predictor_dropout: float = 0.3

    # ── Output ──
    horizon: int = 28  # Predict 28 days


@dataclass
class TrainConfig:
    """Training configuration — optimized for RTX 4050 6GB."""
    # ── Optimization ──
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15  # Early stopping patience
    min_delta: float = 1e-4

    # ── Loss ──
    loss_fn: str = 'tweedie'  # 'tweedie', 'mse', 'huber'
    tweedie_p: float = 1.5    # Tweedie power parameter (1 < p < 2)

    # ── Scheduler ──
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5

    # ── Regularization ──
    gradient_clip: float = 1.0
    label_smoothing: float = 0.0

    # ── Batching — tuned for 6GB VRAM (RTX 4050) ──
    batch_size: int = 256  # Reduced from 512 for 6GB VRAM
    num_workers: int = 2   # Keep low on Windows

    # ── Mixed precision — saves ~40% VRAM ──
    use_amp: bool = True

    # ── Chaos adversarial training ──
    adversarial_training: bool = False
    adversarial_ratio: float = 0.3
    adversarial_epsilon: float = 0.01

    # ── Ensemble ──
    num_ensemble: int = 5

    # ── Checkpointing ──
    checkpoint_dir: str = './checkpoints'
    save_every: int = 10  # Save checkpoint every N epochs
    resume_from: str = ''  # Path to checkpoint to resume from

    # ── GPU settings ──
    pin_memory: bool = True
    cudnn_benchmark: bool = True


@dataclass
class ChaosConfig:
    """Chaos engineering configuration with Hawkes Process support."""
    # ── Perturbation types and severities ──
    demand_shock_severity: float = 0.5
    demand_shock_window: int = 14
    supply_disruption_prob: float = 0.1
    supply_disruption_window: int = 14
    price_volatility_scale: float = 0.3
    calendar_shift_days: int = 3
    graph_corruption_ratio: float = 0.2
    adversarial_epsilon: float = 0.01
    adversarial_steps: int = 5

    # ── Evaluation ──
    num_chaos_trials: int = 10

    # ── Hawkes Process Parameters ──
    use_hawkes: bool = True

    # Parameter grid for adversarial regime search
    hawkes_mu_values: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2]
    )
    hawkes_alpha_values: List[float] = field(
        default_factory=lambda: [0.3, 0.6, 1.0]
    )
    hawkes_beta_values: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0]
    )

    # Default Hawkes config (for quick single-config runs)
    hawkes_default_mu: float = 0.1
    hawkes_default_alpha: float = 0.6
    hawkes_default_beta: float = 1.0

    # Intensity trace storage
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
    experiment_name: str = 'siggnn_v1'

    def __post_init__(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            if self.train.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════
# Preset configurations for different experiment phases
# ═══════════════════════════════════════════════════════════════

def get_debug_config():
    """Minimal config for debugging (100 items, 10 epochs)."""
    cfg = ExperimentConfig()
    cfg.data.max_items = 100
    cfg.train.max_epochs = 10
    cfg.train.num_ensemble = 1
    cfg.model.gat.num_layers = 1
    cfg.chaos.use_hawkes = False  # Skip Hawkes for debug
    cfg.experiment_name = 'debug'
    return cfg


def get_phase1_config():
    """Phase 1: Single store, fast iteration."""
    cfg = ExperimentConfig()
    cfg.data.stores = ['CA_1']
    cfg.train.max_epochs = 60
    cfg.train.num_ensemble = 3
    cfg.experiment_name = 'phase1_CA1'
    return cfg


def get_phase2_config():
    """Phase 2: All CA stores, cross-store learning."""
    cfg = ExperimentConfig()
    cfg.data.stores = ['CA_1', 'CA_2', 'CA_3', 'CA_4']
    cfg.train.max_epochs = 80
    cfg.train.num_ensemble = 5
    cfg.experiment_name = 'phase2_CA'
    return cfg


def get_full_config():
    """Phase 3: Full dataset, all 10 stores."""
    cfg = ExperimentConfig()
    cfg.data.stores = []  # Empty = all stores
    cfg.train.max_epochs = 100
    cfg.train.num_ensemble = 5
    cfg.train.adversarial_training = True
    cfg.experiment_name = 'full_siggnn'
    return cfg


def get_gpu_optimized_config():
    """Config optimized for RTX 4050 6GB VRAM."""
    cfg = ExperimentConfig()
    cfg.data.stores = ['CA_1']
    cfg.train.batch_size = 256
    cfg.train.use_amp = True
    cfg.train.max_epochs = 80
    cfg.train.num_ensemble = 3
    cfg.model.gat.hidden_dim = 96    # Slightly reduced for VRAM
    cfg.model.gat.num_layers = 2     # 2 layers saves VRAM
    cfg.model.predictor_hidden = 192
    cfg.chaos.use_hawkes = True
    # Single Hawkes config for speed
    cfg.chaos.hawkes_mu_values = [0.1]
    cfg.chaos.hawkes_alpha_values = [0.6]
    cfg.chaos.hawkes_beta_values = [1.0]
    cfg.experiment_name = 'gpu_rtx4050'
    return cfg
