# SigGNN Mathematical and Architectural Specification

This document provides the complete mathematical and architectural specification for the SigGNN pipeline. It defines foundational theorems, derivation formulas, discrete implementations, and numerical stability constraints, fully grounded in academic literature.

## 1. Project Directory Structure
```text
m5_siggnn/
├── config.py                 # Centralized configuration (RTX 4050 vs A100 HPC)
├── run_m5.py                 # Evaluation and submission pipeline
├── train.py                  # PyTorch trainer (full-batch, AMP-enabled)
├── diagnose_gnn.py           # Layer stability and residual tracing diagnostics
├── main.py                   # Alternate trainer integration for diagnostics
├── a100_lightning_guide.md   # Deployment guide for Lightning AI A100
├── PROJECT_DOCUMENTATION.md  # This document
│
├── models/                   # Core Architecture
│   ├── siggnn.py             # Model definitions, predictors, custom losses
│   ├── signature.py          # Multi-scale signature engine (iterated integrals)
│   ├── gat.py                # Sparse temporal graph attention networks
│   └── reconciliation.py     # Hierarchical constraint mapping (MinT)
│
├── data/                     # Data Pipeline
│   ├── loader.py             # M5 dataset parser
│   ├── features.py           # Feature extraction (lags, rolling stats, momentum)
│   ├── graph_builder.py      # Hierarchy graph construction (store/dept/category)
│   └── wrmsse.py             # Official M5 metric implementation
│
├── chaos/                    # Resilience Testing
│   ├── engine.py             # Stress-test controller
│   ├── hawkes_process.py     # Contagion probability modeling
│   └── perturbations.py      # Graph dropout and adversarial corruptions
│
├── dataset/                  # Raw CSVs (Kaggle CLI download)
├── checkpoints/              # Model checkpoints and gradient histories
└── experiments/              # Hawkes process intensity logs
```

## 2. Multi-Scale Signature Engineering
**Module:** `models/signature.py`

Path signatures serve as universal feature extractors for time-series data, capturing geometric structure and cross-channel interactions independent of the sampling rate.

**References:**
- Kidger, P., et al. (2019). Deep Signature Transforms. NeurIPS.
- Chevyrev, I., & Kormilitzin, A. (2016). A Primer on the Signature Method in Machine Learning. arXiv:1603.03788.

### 2.1 Continuous Formulation and Chen's Identity
Model a retail time series as a continuous path of bounded variation, $X: [0, T] \to \mathbb{R}^d$. The truncated signature $S(X)$ to depth $m$ is the tensor series:

$$S(X)_{0,T} = \left( 1, \mathbf{X}^1, \mathbf{X}^2, \ldots, \mathbf{X}^m \right)$$

where the $k$-th level is the iterated integral:

$$\mathbf{X}^k = \int_{0 < t_1 < \cdots < t_k < T} dX_{t_1} \otimes \cdots \otimes dX_{t_k} \in (\mathbb{R}^d)^{\otimes k}$$

Chen's Identity enables efficient computation over rolling windows. For $s \le t \le u$:

$$S(X)_{s,u} = S(X)_{s,t} \otimes S(X)_{t,u}$$

This allows SigGNN to compute signatures over overlapping windows (7, 14, 28 days) without redundant global integration.

### 2.2 Discrete Implementation
For a discrete series $X = (X_1, \ldots, X_M)$ with increments $\Delta X_t = X_{t+1} - X_t$, the framework computes depth $m=2$:

**Level 1 (Path Increments):**
$$\mathbf{X}^1 = \sum_{t=1}^{M-1} \Delta X_t \in \mathbb{R}^d$$

**Level 2 (Cross-Area Tensor):**
$$\mathbf{X}^2 = \sum_{t=2}^{M-1} \Delta X_t \otimes \left( \sum_{s=1}^{t-1} \Delta X_s \right) \in \mathbb{R}^{d \times d}$$

**Numerical Stability:** Computing $\mathbf{X}^2$ requires $\mathcal{O}(d^2)$ operations via einsum. For $d=26$ features over $T=90$ frames, accumulated values can exceed FP16 limits (65,504).
- Force FP32 precision during signature computation (`path.float()`).
- Clamp outputs to the range `[-50, 50]`.

### 2.3 Lead-Lag Augmentation
Standard signatures lack explicit time parameterization. To capture quadratic variation—critical for price momentum—we apply a lead-lag transformation. Given $X = (x_1, \ldots, x_M)$, construct $X_{LL} \in \mathbb{R}^{(2M-1) \times 2d}$:

$$X_{LL} = \big[ (x_1, x_1), (x_1, x_2), (x_2, x_2), (x_2, x_3), \ldots, (x_M, x_M) \big]$$

Computing signatures on this augmented path yields terms of the form $\int X_{\text{lead}} \otimes dX_{\text{lag}}$, explicitly encoding the autocovariance structure.

## 3. Sparse Temporal Graph Attention
**Module:** `models/gat.py`

SigGNN models supply chain dependencies using graph attention networks that respect hierarchical structure (Store → Category → Item).

**References:**
- Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
- Brody, S., et al. (2022). How Attentive are Graph Attention Networks? ICLR.

### 3.1 Edge-Conditional Multi-Head Attention
For node $i$ at layer $l$ with embedding $h_i^{(l)} \in \mathbb{R}^F$, attention head $k$ computes:

$$q_i^{(k)} = W_Q^{(k)} h_i, \quad k_j^{(k)} = W_K^{(k)} h_j, \quad v_j^{(k)} = W_V^{(k)} h_j$$

Unlike static GAT, we incorporate temporal edge embeddings $E_{\text{edge}}(t_{ij})$:

$$e_{ij}^{(k)} = \text{LeakyReLU}_{0.2} \left( a^{(k)\top} \big[ q_i^{(k)} \parallel k_j^{(k)} \big] \right) + q_i^{(k)} \cdot E_{\text{edge}}(t_{ij})$$

### 3.2 Stabilized Sparse Softmax
To prevent gradient collapse in densely connected regions, attention weights use a numerically stable softmax:

$$\alpha_{ij}^{(k)} = \frac{\exp(e_{ij}^{(k)} - \epsilon_i)}{\sum_{l \in \mathcal{N}(i)} \exp(e_{il}^{(k)} - \epsilon_i) + 10^{-6}}$$

where $\epsilon_i = \max_l(e_{il}^{(k)})$ prevents overflow.

**Edge Case:** For isolated nodes, $\epsilon_i \to -\infty$. We enforce $\epsilon_i \ge -100$.

The layer update with residual connection and normalization is defined as:

$$h_i^{(l+1)} = \text{LayerNorm}\left( h_i^{(l)} + \bigoplus_{k=1}^{H} \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} v_j^{(k)} \right) \right)$$

## 4. Loss Functions and Distributional Assumptions
**Module:** `models/siggnn.py`

**References:**
- Makridakis, S., et al. (2022). The M5 accuracy competition. International Journal of Forecasting.
- Tweedie, M. C. K. (1984). An index which distinguishes between some important exponential families.

### 4.1 WRMSSE Surrogate Loss
The M5 competition metric is Weighted Root Mean Squared Scaled Error:

$$\text{WRMSSE} = \sum_{i=1}^{N} \omega_i \sqrt{\frac{\frac{1}{H} \sum_{t=1}^{H} (Y_{i,t} - \hat{Y}_{i,t})^2}{\frac{1}{M-1} \sum_{t=2}^{M} (Y_{i,t} - Y_{i,t-1})^2}}$$

The denominator $s_i^2$ represents naive forecast variance (random walk baseline). Direct optimization of the square root is intractable; we use a proportional MSE surrogate:

$$\mathcal{L}_{\text{WRMSSE}} = \sum_i \frac{\omega_i}{\max(1, s_i)^2} \cdot \text{MSE}_i$$

**Stability Constraints:**
- Floor $s_i$ at 1.0 to prevent division by zero for flat series.
- Clip outlier weights: $\omega_i / s_i^2 \le 10 \times \text{median}$.

### 4.2 Tweedie Deviance Loss
Retail demand exhibits point masses at zero with continuous positive support—characteristic of Tweedie distributions. For power parameter $p = 1.5$ (compound Poisson-Gamma):

$$\mathcal{L}_{\text{Tweedie}} = \sum_i \left( -Y_i \frac{\hat{\mu}_i^{1-p}}{1-p} + \frac{\hat{\mu}_i^{2-p}}{2-p} \right)$$

**Numerical Stability:** The term $\hat{\mu}^{-0.5}$ diverges as $\hat{\mu} \to 0$. Predictions are clamped to the range `[0.0001, 1,000,000]`.

## 5. Hierarchical Reconciliation (MinT)
**Module:** `models/reconciliation.py`

Forecasts must aggregate coherently across the hierarchy (Item → Department → Store → Total).

**Reference:** Wickramasuriya, S. L., et al. (2019). Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization. JASA.

### 5.1 Optimal Projection
The Minimum Trace reconciliation projects base forecasts $\hat{Y}$ onto the coherent subspace:

$$\tilde{Y} = S P \hat{Y}, \quad \text{where} \quad P = (S^\top W S)^{-1} S^\top W$$

Here $S \in \mathbb{R}^{\text{total} \times N}$ is the summation matrix encoding hierarchy constraints.

### 5.2 GPU-Efficient Approximation
Computing $P$ exactly requires $\mathcal{O}(N^3)$ matrix inversion. We approximate with learnable diagonal scaling:

$$\hat{Y}_{\text{reconciled}} = \min\big( (Y_\mu + 1) \times 20, \text{ReLU}(\hat{Y} \cdot \text{Scale}_g + \beta_g) \big)$$

This maintains coherence while avoiding expensive matrix operations on constrained GPUs.

## 6. Pipeline Architecture and Implementation Details

### 6.1 Evaluation Pipeline (`run_m5.py`)
- **Post-hoc Calibration:** After generating ensemble predictions $\hat{Y}_{\text{blend}} = \alpha \cdot \hat{Y}_{\text{model}} + (1-\alpha) \cdot \hat{Y}_{\text{DOW}}$, day-of-week and item-level multipliers are fit on validation data:
  $$\text{multiplier} = \frac{\sum Y_{\text{actual}} \cdot \hat{Y}_{\text{val}}}{\sum \hat{Y}_{\text{val}}^2}$$
  This calibration step reduces WRMSSE to approximately 0.50.
- **Feature Window:** A 140-day buffer ensures rolling statistics (up to 56-day windows on top of 84-day lags) compute without NaN propagation.
- **Temporal Augmentation:** Five overlapping windows ($t, t-28, t-56, \ldots$) prevent the model from exploiting absolute date artifacts.

### 6.2 Training Loop (`train.py`)
- **Full-Batch Constraint:** Mini-batching would fragment the hierarchy graph, disconnecting items from their aggregation structure. The entire graph processes in a single forward pass.
- **Learning Rate Schedule:** `CosineAnnealingWarmRestarts` with $T_0 = 20$ epochs. Periodic learning rate resets help escape local minima around WRMSSE 0.68–0.70.
- **Gradient Triage:** Real-time NaN/Inf detection. After 3 consecutive failures, the learning rate is halved to preserve training progress on long A100 runs.

### 6.3 Stress Testing (`chaos/engine.py`)
- **Hawkes Process Contagion:** Supply chain disruptions are modeled as self-exciting point processes with parameters $(\mu, \alpha, \beta)$. Stability requires $\alpha / \beta < 1$ to prevent unbounded intensity.
- **Graph Dropout:** 10% to 30% of edges are randomly removed to simulate missing data or broken supply links, testing the model's robustness to sparse connectivity.

### 6.4 Diagnostics (`diagnose_gnn.py`)
- **Validation Criterion:** Model predictions $\hat{Z}$ must satisfy:
  $$\text{Std}(\hat{Z}) \propto \text{Std}(Y_{\text{residual}})$$
  If prediction variance is disproportionate to residual variance, the model is learning noise rather than demand structure.

## 7. Development History and Hardware Evolution

### 7.1 Phase I: Consumer Hardware (v1.0–1.5)
- **Environment:** NVIDIA RTX 4050, 6GB VRAM
- **Challenges:**
  - Unbounded attention scores caused gradient explosion (WRMSSE > 3.0).
  - 90-day signature paths exceeded memory limits.
  - FP16 overflow during signature tensor products.
- **Solutions:**
  - Clamped all tensor values to `[-50, 50]`.
  - Forced FP32 for signature computation.
  - Reduced to single-store processing.

### 7.2 Phase II: Algorithmic Refinement (v2.0–2.5)
- **Environment:** RTX 4050 (continued)
- **Challenges:**
  - Promotional spikes collapsed to mean predictions.
  - Zero-variance items produced infinite WRMSSE weights.
- **Solutions:**
  - Shifted to residual learning: $\hat{Y} = \text{ReLU}(\text{Baseline} + Z)$.
  - Introduced $\max(1, s_i)$ floor in scaling denominators.
  - WRMSSE improved from >0.85 to <0.70.

### 7.3 Phase III: Cloud Scaling (v3.0)
- **Environment:** Lightning AI (NVIDIA A100, 40–80GB)
- **Deployment Considerations:**
  - Dataset transfer via Kaggle competitions download (CLI).
  - Feature generation requires 16+ CPUs and 64GB RAM to avoid bottlenecks.
  - Use `tmux` for persistent sessions during 45+ minute preprocessing.
- **Capability Gains:** The `--a100` flag enables:
  - Full 90-day signature windows.
  - 3 GAT layers with 8 attention heads.
  - Target WRMSSE $\approx$ 0.65.
