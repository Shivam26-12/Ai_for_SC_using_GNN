print("--- SCRIPT IS STARTING ---")
import sys
print(f"Python Path: {sys.path}")
import torch
import numpy as np
import pandas as pd
from config import get_debug_config
from data.loader import M5DataLoader
from data.features import FeatureEngineer

def run_diagnostic():
    print("🔍 --- STARTING DATA DIAGNOSTIC ---")
    config = get_debug_config()
    
    # 1. Check Raw Files
    print("\nStep 1: Checking Raw CSVs...")
    for path, name in [(config.data.sales_path, "Sales"), 
                       (config.data.prices_path, "Prices"), 
                       (config.data.calendar_path, "Calendar")]:
        df = pd.read_csv(path)
        nan_count = df.isnull().sum().sum()
        print(f"  - {name} CSV: {nan_count} NaNs found.")
        if nan_count > 0:
            print(f"    ⚠️ NaNs in {name} are in columns: {df.columns[df.isnull().any()].tolist()}")

    # 2. Check DataLoader Output
    print("\nStep 2: Checking DataLoader Output...")
    loader = M5DataLoader(config.data)
    dataset = loader.prepare_dataset()
    
    for key in ['sales_matrix', 'price_matrix', 'calendar_features']:
        data = dataset[key]
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        print(f"  - {key}: NaNs? {has_nan} | Infs? {has_inf}")
        if has_nan:
            # Find which row/item is failing
            row_idx = np.where(np.isnan(data).any(axis=1))[0]
            print(f"    ❌ NaN detected in {len(row_idx)} rows. First failing index: {row_idx[0]}")

    # 3. Check Feature Engineering
    print("\nStep 3: Checking Feature Engineering...")
    fe = FeatureEngineer(config.data, config.features)
    
    # Test a small window
    train_data = fe.build_stream_tensors(
        dataset, 
        start_day=1000, 
        end_day=1100,
        device=torch.device('cpu')
    )
    
    node_feats = train_data['node_features'].numpy()
    targets = train_data['targets'].numpy()
    
    print(f"  - Node Features (Final Tensor): NaNs? {np.isnan(node_feats).any()}")
    print(f"  - Targets: NaNs? {np.isnan(targets).any()}")
    
    if np.isnan(node_feats).any():
        # Identify which specific feature channel is the culprit
        # (e.g., Log demand, Lags, Rolling, Prices, Calendar)
        for c in range(node_feats.shape[-1]):
            if np.isnan(node_feats[:, :, c]).any():
                print(f"    💥 Feature Channel {c} contains NaNs!")

if __name__ == "__main__":
    run_diagnostic()