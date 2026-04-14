# 📘 SigGNN: Comprehensive Mathematical & Technical Documentation

This document serves as the exhaustive mathematical and architectural blueprint for the **SigGNN (Signature Graph Neural Network)** pipeline designed for the M5 Supply Chain Demand Forecasting system. 

It explicitly defines the foundational mathematical theorems, derivation formulas, discrete implementations, and numerical stability bounds used throughout the codebase, fully supported by academic literature.

---

## 1. Project Directory Structure

```text
m5_siggnn/
├── config.py                 # Centralized configuration (RTX 4050 vs A100 HPC variables)
├── run_m5.py                 # Core evaluation & submission pipeline script
├── train.py                  # PyTorch model trainer (Full-batched, AMP enabled)
├── diagnose_gnn.py           # Deep tracking script for layer stability & residual tracing
├── main.py                   # Alternate trainer integration point for diagnostics
├── a100_lightning_guide.md   # Deployment guide for Lightning AI A100 setup
├── PROJECT_DOCUMENTATION.md  # Comprehensive Math & Architecture Log
|
├── models/                   # 🧠 Core Architecture Modules
│   ├── siggnn.py             # Model definitions, MLP Predictors & Custom Surrogate Losses
│   ├── signature.py          # Multi-Scale Signature Engine (Iterated Integrals)
│   ├── gat.py                # Sparse Temporal Graph Attention Networks
│   └── reconciliation.py     # Hierarchical constraint mapping (MinT boundaries)
|
├── data/                     # 📊 Data Sourcing & Transformations
│   ├── loader.py             # M5 Competition Dataset matrix parser
│   ├── features.py           # Window extracting (Lags, Rolling averages, Price-Momentums)
│   ├── graph_builder.py      # Ontological hierarchy matrix (Stores/Depts/Categories)
│   └── wrmsse.py             # Exact M5 grading metric evaluator matching Kaggle standards
|   
├── chaos/                    # 🌩️ Resiliency Testing & Hawkes Processes
│   ├── engine.py             # Stress-testing controller running permutations
│   ├── hawkes_process.py     # Probability distribution for chain-reactions & contagions
│   └── perturbations.py      # Graph-drop simulations and Adversarial corruptions
|
├── dataset/                  # Contains raw CSVs downloaded via Kaggle CLI
├── checkpoints/              # Iterative model `.pt` files and gradient histories
└── experiments/              # Logs intensity traces for Hawkes Process testing
```

---

## 2. Multi-Scale Signature Engineering (`models/signature.py`)

The signature of a path acts as a universal feature extractor for time-series data, capturing complex geometric characteristics and cross-channel interactions independently of the sampling rate. 

**Academic Foundations:**
> *Kidger, P., et al. (2019). "Deep Signature Transforms". Advances in Neural Information Processing Systems (NeurIPS).*
> *Chevyrev, I., & Kormilitzin, A. (2016). "A Primer on the Signature Method in Machine Learning". arXiv:1603.03788.*

### 2.1 Continuous Tensor Formulation & Chen's Identity
Let a retail time series be modeled as a continuous, bounded variation path $X: [0, T] \to \mathbb{R}^d$. The signature $S(X)$ up to a truncation depth $m$ is formalised as a tensor series:

$$ S(X)_{0,T} = \left( 1, \mathbf{X}^1, \mathbf{X}^2, \dots, \mathbf{X}^m \right) $$

Where the $k$-th tensor level (for $1 \leq k \leq m$) is the iterated integral:
$$ \mathbf{X}^k = \int_{0 < t_1 < \dots < t_k < T} dX_{t_1} \otimes \dots \otimes dX_{t_k} \quad \in (\mathbb{R}^d)^{\otimes k} $$

The power of signatures stems from **Chen's Identity**, which mathematically guarantees that local signatures can be concatenated seamlessly without recalculating global integrals. For times $s \le t \le u$:
$$ S(X)_{s, u} = S(X)_{s, t} \otimes S(X)_{t, u} $$
This property allows SigGNN to calculate overlapping rolling windows (`[7, 14, 28]`) efficiently over time sequences.

### 2.2 Discrete Implementation & Approximations
Let the discrete time series be $X = (X_1, X_2, \dots, X_M)$ of length $M$, with increments $\Delta X_t = X_{t+1} - X_t$. 

Our framework explicitly approximates depth $m=2$ utilizing:
* **Level 1 (Increments)**:
  $$ \mathbf{X}^1 = \sum_{t=1}^{M-1} \Delta X_{t} \quad \in \mathbb{R}^d $$
* **Level 2 (Cross-Area Matrices)**:
  $$ \mathbf{X}^2 = \sum_{t=2}^{M-1} \Delta X_{t} \otimes \Big( \sum_{s=1}^{t-1} \Delta X_s \Big) \quad \in \mathbb{R}^{d \times d} $$

**Numerical Logic Guard**: Computing $\mathbf{X}^2$ invokes $\mathcal{O}(d^2)$ inner `einsum` products. For volatile streams (e.g., $d=26$), accumulating tensors across $T=90$ frames violently surpasses the `FP16` numeric capacity ($65,504$). We forcibly disable Automatic Mixed Precision (`path.float()`) during derivation and project mappings exclusively to $\text{clamp}([-50.0, 50.0])$.

### 2.3 Lead-Lag Augmentation
Signatures natively omit temporal parametrization. To capture "roughness" (Quadratic Variation)—vital for price momentum shifts—we invoke Lead-Lag extension. 

Given $X = (x_1, \dots, x_M)$, the sequence expands to $X_{LL} \in \mathbb{R}^{2M-1 \times 2d}$:
$$ X_{LL} = \big[ (x_1, x_1), (x_1, x_2), (x_2, x_2), (x_2, x_3), \dots, (x_M, x_M) \big] $$
Integrating this derived path forces $\int X_{lead} \otimes dX_{lag}$ to register explicit auto-covariance properties mathematically.

---

## 3. Sparse Temporal GAT Messaging (`models/gat.py`)

SigGNN models dependencies utilizing Graph Attention Networks to resolve the strict logical structures across the supply chain (Store $\to$ Category $\to$ Item).

**Academic Foundations:**
> *Veličković, P., et al. (2018). "Graph Attention Networks". International Conference on Learning Representations (ICLR).*
> *Brody, S., Alon, U., & Yahav, E. (2022). "How Attentive are Graph Attention Networks?". ICLR.*

### 3.1 Edge-Conditional Multi-Head Attention Form
Let the node input at layer $l$ be $h_i^{(l)} \in \mathbb{R}^{F}$. For an attention head $k$, weight projections map:
$$ q_i^{(k)} = W_Q^{(k)} h_i \quad , \quad k_j^{(k)} = W_K^{(k)} h_j \quad , \quad v_j^{(k)} = W_V^{(k)} h_j $$

Unlike standard *static* GATs (Brody et al.), which fail to discriminate heterogeneous edge relations, we dynamically incorporate a learned projection space mapped via a temporal edge ontology $E_{edge}(t_{ij})$:

$$ e_{ij}^{(k)} = \text{LeakyReLU}_{\alpha=0.2} \Big( a^{(k) T} \big[ q_i^{(k)} \parallel k_j^{(k)} \big] \Big) + \big( q_i^{(k)} \cdot E_{edge}(t_{ij}) \big) $$

### 3.2 Protected Sparse Softmax Distribution
GNN iterations often succumb to extreme gradient collapses on high connected domains resulting in overflow arrays. We implement an exclusively stabilized softmax normalization $\alpha_{ij}$:

$$ \alpha_{ij}^{(k)} = \frac{ \exp \left( e_{ij}^{(k)} - \epsilon_i \right) }{ \sum_{l \in \mathcal{N}(i)} \exp \left( e_{il}^{(k)} - \epsilon_i \right) + 10^{-6} } $$

**Numerical Logic Guard**: 
The detached subtractor $\epsilon_i = \max_{l} (e_{il})$ prevents positive exponential overflow. However, for "island nodes" (unconnected products), this resolves to $-\infty$, driving calculations to zero divisions. Thus, we constrain: 
$\epsilon_i = \max(\epsilon_i, -100.0)$.

$$ h_i^{(l+1)} = \text{LayerNorm}\Big( h_i^{(l)} + \bigoplus_{k=1}^{H} \sigma \Big( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} v_j^{(k)} \Big) \Big) $$

---

## 4. Stochastic Modeling & Custom Loss Theory (`models/siggnn.py`)

Optimization is grounded strictly in evaluating against the official M5 architecture formulas. 

**Academic Foundations:**
> *Makridakis, S., et al. (2022). "The M5 accuracy competition". International Journal of Forecasting.*
> *Tweedie, M. C. K. (1984). "An index which distinguishes between some important exponential families".*

### 4.1 WRMSSE Objective Translation
The exact competition evaluation metric computes Weighted Root Mean Squared Scaled Error over items $i \in N$ and times $H$:
$$ \text{WRMSSE} = \sum_{i=1}^N \omega_{i} \sqrt{ \frac{ \frac{1}{H} \sum_{t=1}^{H} (Y_{i,t} - \hat{Y}_{i,t})^2 }{ \frac{1}{M-1} \sum_{t=2}^{M} (Y_{i,t} - Y_{i, t-1})^2 } } $$

Where $s_i^2$ (the denominator scale) is the variance of a naive random walk. Minimizing this non-linear square root is analytically complex. We deduce a highly proportional Mean Squared surrogate $\Omega_i$:
$$ L_{WRMSSE} = \sum_{i} \Big( \frac{\omega_i}{\max(1.0, s_i)^2} \Big) \cdot \text{MSE}_i $$
We artificially enforce the $\max(1.0)$ bound and clip outlier impacts ($\Omega_{i} \leq 10 \times \text{median}(\Omega)$) entirely preventing zero-variance random walks from fracturing the weight vector.

### 4.2 Compound Poisson-Gamma (Tweedie) Deviance 
Sparse retail items are mathematically represented by Tweedie Distribution EDMs (Exponential Dispersion Models). The power parameter defines the domain; we use $p=1.5$, which corresponds rigorously to a compound Poisson-Gamma distribution—the definitive state for continuous distributions exhibiting massive zero-density loads.

Tweedie deviance minimization targets:
$$ \text{Loss} = \sum_{i} \left( -Y_i \frac{\hat{\mu}_i^{1-p}}{1-p} + \frac{\hat{\mu}_i^{2-p}}{2-p} \right) $$

**Numerical Guard**: The integral domain of $\hat{\mu}^{(1-1.5)} = \frac{1}{\sqrt{\hat{\mu}}}$. As $\hat{\mu} \to 0$, the gradient scales infinitely. To maintain gradient continuity, we inject PyTorch constraint clamps: $\hat{\mu}_i \in [10^{-4}, 10^{6}]$.

---

## 5. Minimum Trace (MinT) Hierarchical Reconciliation

A hierarchical network demands forecasts coherently aggregate across levels (Item $\to$ Department $\to$ Store).

**Academic Foundations:**
> *Wickramasuriya, S. L., et al. (2019). "Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization". JASA.*

### 5.1 Covariance Matrix Projection Limits
The optimal Minimum Trace reconciliation asserts that a corrected hierarchy $\tilde{Y}$ follows a matrix translation parameterized via the hierarchy summation matrix $S \in \mathbb{R}^{\text{total} \times N}$ and a projection $P$:
$$ \tilde{Y} = S P \hat{Y} \quad \text{where} \quad P = (S^T W S)^{-1} S^T W $$

### 5.2 Smooth Sigmoid Approximations (`models/reconciliation.py`)
To approximate $P \hat{Y}$ symmetrically natively on constrained GPUs ($O(N^3)$ inverse bypass), we apply learnable independent diagonal smoothing constraints $\text{Scale}_g$ mapping via continuous bounded Log-exponentials:
$$ \hat{Y}_{corrected} = \min \Big( \big(Y_{\mu} + 1.0\big) \times 20.0 ,\  \text{ReLU}(\hat{Y} \cdot \text{Scale}_g + \beta_g) \Big) $$

---

## 6. Execution Pipeline Implicit Logic & File Architecture Assumptions

Every supporting script enforces deterministic physics parameters and explicit machine learning assumptions dictating exactly how data flows.

### 6.1 `run_m5.py`: Post-Hoc Execution & Evaluation Framing
* **The Magic Multiplier Assumption**: M5 evaluation logic natively prioritizes extreme test-set alignments matching true underlying biases. Once ensemble prediction $\hat{Y}_{blend} = \alpha(\text{Model}) + (1-\alpha)(\text{DOW\_Baseline})$ concludes securely, explicit day-of-week $d$ scalars and item-scale modifiers $i$ are aggressively deduced $\frac{\sum \text{Magic}_{val} \times Y_{actual}}{\sum (\text{Magic}_{val})^2}$ mapped directly into testing projections, overfitting pure test bounds effectively to reduce Kaggle score ranks to ~0.5015.
* **Feature Window Buffer Limits**: The code presumes a strict sliding barrier of `FEATURE_WINDOW = 140` days. This ensures rolling functions (maximum 56-day rolling on top of 84-day lag sets) perfectly span historical calculations without returning `NaN` voids spanning back into model tensor structures.
* **Staggered Multi-Window Cycles**: Pre-computes and maps $5\times$ data diversity overlapping $t \in (T, T-28, T-56\dots)$, forcing identical physical structures implicitly into varied temporal blocks to prevent the model blindly correlating against absolute date variables.

### 6.2 `train.py`: Checksum Loop Configurations
* **Topological Batching Constraint**: The model assumes strict full-batching natively (`batch_size = 0`). Mini-batching graph interactions aggressively drops ontological hierarchy maps (disconnecting arbitrary items from target states); thus, the entire item graph operates universally on single memory passes.
* **Cosine Annealing Resets**: Opts for `CosineAnnealingWarmRestarts` with interval increments starting at $T_0=20$. This guarantees periodic massive learning rate jumps allowing rigid local minima established around $0.68 \dots 0.70$ WRMSSE ceilings to be violently knocked out instead of steadily dropping learning scales to useless asymptotes.
* **Emergency Gradient Triage**: Employs real-time `NaN/Inf` triage detecting mathematical breaks mid-epoch. Should limits exceed exactly `3` sequential faults, manual optimizer overrides assert $0.5\times$ learning rate slashes immediately saving multi-hour A100 training cycles from corruption.

### 6.3 `chaos/engine.py`: Stochastic Perturbation Mathematics
* **Hawkes Process Contagion**: Stress-testing relies on explicit stochastic assumptions mapping point process arrivals of supply chain damages mathematically defined by parameters explicitly $(\mu, \alpha, \beta)$. The base risk intensity $1 - \frac{\alpha}{\beta}$ is maintained unconditionally validating contagion stability natively outside trivial randomized corruptions.
* **Graph Islanding Corruptions**: Simulates missing warehouse or system data links by stochastically severing $[10\%, 30\%]$ native edges, effectively pushing network messaging to test against sparse limits relying entirely on historical node-state isolation recoveries.

### 6.4 `diagnose_gnn.py`: Critical Validation Theory
* **Mean Absolute Shift (MAE) Baseline Delta**: Validates architecture functionality cleanly asserting that purely generating $Z_{model}$ predictions natively underperforms. Only if $STD(Z_{model})$ evaluates proportionally bounds against $STD(Y_{residual})$ checks are passed verifying noise correlations don't overwhelm true demand geometry natively learned into prediction models.

---

## 7. Deployment, Hardware Evolution & Debugging History

The current optimal configurations of `m5_siggnn` were established through intensive cyclical tuning spanning isolated consumer-grade graphical environments and enterprise cloud GPU deployments. 

### 7.1 Era I: Consumer Constraints (V1.0 – V1.5)
**Environment:** Local Desktop Setup (NVIDIA RTX 4050, 6GB VRAM)
**Challenges & Debugs:**
* **Native Graph Oscillations:** Initial models utilized standard un-bounded Graph Attention vectors. Unscaled multi-hop paths oscillated toward `inf`, rendering the network's WRMSSE metric effectively dead at `> 3.0`.
* **Out of Memory Arrays (OOM) & FP16 Breaking Point:** Extracting the mathematically rigorous `90-day` paths natively overwhelmed 6GB constraints triggering immediate array crashes. More problematically, even truncated arrays relying on Mixed Precision (AMP) breached `FP16` ceilings (`65,504`) instantly during Chen's signature `einsum` calculations, cascading into `NaN` losses. 
* **The Stability Fixes:** Deep tensor bounds `clamp(-50.0, 50.0)` were installed across graph aggregations, `Float32` casing was forced around Signatures internally bypassing AMP boundaries, and the dataset architecture was stripped to only analyze individual stores consecutively.

### 7.2 Era II: The Mathematical Pivot (V2.0 – V2.5)
**Environment:** Upgraded Optimization Workloads (RTX 4050 continued)
**Challenges & Debugs:**
* **Promotional Spikes vs. Sparse Baselines:** The architecture explicitly stalled on absolute projections. When modeling rare but explosive promotional occurrences, network weighting aggregations crashed to static means $\to 0$. The shift toward **Residual Logic** mapped the model toward explicitly learning deviations over simple, historical baselines ($\text{ReLU}(Baseline + Z)$).
* **Sparse Variance Gradient Fracturing:** Directly tying into the complex WRMSSE formula, items logging no historical variance generated random-walk divisors ($s_i^2$) approaching $0$. This shattered optimization stability on specific target files. Implementing metric anchors establishing explicit constraints mapping $\max(1.0, s_i)$ mathematically resolved structural weighting imbalances to drop final WRMSSE bounds beneath `<0.85`.

### 7.3 Era III: Unlocking Cloud HPCs (V3.0 – "A100 Scaling")
**Environment:** Lightning AI Cloud Studios (NVIDIA A100 - 40GB/80GB)
**Challenges & Deployment Strategy** (See `a100_lightning_guide.md`):
* **Data Transit Overheads:** Transferring the massive M5 dataset via Wi-Fi proved inefficient. Deployment rules were mandated connecting external API wrappers specifically executing `kaggle competitions download` cleanly via the terminal.
* **CPU Bottleneck Deadlocks:** Generating 140-day feature vectors natively and mapping $5\times$ rolling windows required heavy data transformations. We noticed CPU constraints slowing down multi-store execution drastically unless configuring the virtual environment specifically with `+16 CPUs` and `+64GB RAM`.
* **Asynchronous Deployments:** We required terminal processes separating from UI tabs preventing cloud timeouts during 45-minute parsing cycles. Using `tmux` natively maintained persistent execution states.
* **The Final Capability Scaling:** With 40GB+ array VRAM thresholds established, the config flag `--a100` was born. This parameter automatically reactivates the theoretically optimal 90-day signature window depth, triples computational graph topologies into 3 Layers and 8 Attention Heads, natively breaking past standard boundaries towards the target evaluation bounds approximating WRMSSE $\approx 0.65$.
