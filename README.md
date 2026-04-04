# SigGNN for M5 Supply Chain Forecasting 🚀

SigGNN is a research-grade supply chain demand forecasting pipeline. Designed explicitly for the [M5 Forecasting Competition](https://mofc.unic.ac.cy/m5-competition/), it employs graph neural networks and rough path signatures to capture the highly complex spatial (cross-item) and temporal geometry of retail sales data.

## Architecture

1. **Multi-Scale Path Signatures**: We encode time series paths (demand, price, date features) into truncated path signatures at multiple scales (weekly, monthly, quarterly). Lead-lag augmentation allows the encoder to capture volatility and quadratic variation in sales.
2. **Heterogeneous Sparse Temporal GAT**: The graph natively models item-department hierarchy, in-store demand cross-correlations, and cross-store learning for identical items.
3. **Hierarchical Reconciliation**: Learned adjustments ensure coherence at the department and category levels, optimizing for bottom-up inference structures.
4. **Tweedie Deviance Loss**: Handles the zero-inflated, long-tail sales distribution accurately (matching the M5 1st-place solution).
5. **Chaos Engineering & Adversarial Testing**: Includes built-in mechanisms (FGSM/PGD attacks, demand shocks, price volatility spikes) to rigorously benchmark the robustness of the forecasts.

## Getting Started

### 1. Requirements

Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

### 2. Dataset

Download the official dataset from the Kaggle [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) competition. Place the three main CSV files in the `dataset/` directory:

```text
m5_siggnn/
└── dataset/
    ├── sales_train_evaluation.csv
    ├── calendar.csv
    └── sell_prices.csv
```

### 3. Running the Pipeline

To run the complete pipeline which constructs the dataset, trains the SigGNN, evaluates against proper temporal splits, and generates `submission.csv`:

**Run on a single store (e.g., CA_1 - fast, good for testing):**
```bash
python run_m5.py --store CA_1 --epochs 60
```

**Run on the full M5 dataset (all 10 stores):**
```bash
python run_m5.py --store all --epochs 100
```
*(Note: Running the full store mode will take a few hours depending on your GPU).*

## How It Works

The `run_m5.py` orchestrates the complete workflow:
1. **Vectorized Loading**: Loads 6.8 million prices and 30,490 sales series via optimized lookups.
2. **Temporal Validation Split**:
   - Training Data: `d_1` to `d_1885`
   - Internal Validation Data (for WRMSSE): `d_1886` to `d_1913`
   - Test Target (Validation rows): `d_1914` to `d_1941`
   - Final Evaluation Target: `d_1942` to `d_1969`
3. **Graph Construction**: Constructs hierarchical edges, top-*k* correlation edges, and cross-store edges.
4. **WRMSSE Calculation**: Implements the official weighted scaled hierarchical M5 metric for proper checkpoint evaluation.

## Chaos Engineering Overview

The `chaos/` directory contains tools to evaluate how robust the GNN is to real-world interruptions:
- **Demand Shocks:** Random demand spikes (e.g., viral products) or crashes (e.g., lockdowns).
- **Supply Disruptions:** Temporary out-of-stock scenarios.
- **Graph Corruptions:** Simulated loss of access to specific item or store data.
- **Adversarial (PGD/FGSM):** Mathematically optimal gradient perturbations.

Use the `main.py` script for synthetic stress-testing and chaos experimentation:
```bash
python main.py --items 100 --epochs 20
```

## Hardware Setup

The model uses highly optimized sparse structures and avoids O(N²) operations. 
The entire graph fits comfortably in **6GB VRAM** (e.g., RTX 4050 or RTX 2060). For systems with no GPU, training is fully supported on CPU but will be slower.
