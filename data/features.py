"""
Feature Engineering Pipeline — Matches and exceeds M5 winner features.
Produces the tensors that feed into the SigGNN model.
UPDATED: 
- Added per-feature standardization to prevent scale mismatch
- Added DOW baseline computation for residual learning (key M5 winner technique)
"""
import numpy as np
import torch
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig, FeatureConfig


class FeatureEngineer:
    """
    Builds all features needed for the SigGNN model.
    """

    def __init__(self, data_config: DataConfig, feature_config: FeatureConfig):
        self.data_cfg = data_config
        self.feat_cfg = feature_config

    def compute_lag_features(
        self, 
        sales_matrix: np.ndarray, 
        lags: List[int]
    ) -> np.ndarray:
        N, T = sales_matrix.shape
        lag_features = np.zeros((N, T, len(lags)), dtype=np.float32)

        for i, lag in enumerate(lags):
            if lag < T:
                lag_features[:, lag:, i] = sales_matrix[:, :T - lag]
        
        # Guard: Replace any NaNs that might have sneaked in
        return np.nan_to_num(lag_features)

    def compute_rolling_features(
        self, 
        sales_matrix: np.ndarray, 
        windows: List[int]
    ) -> np.ndarray:
        N, T = sales_matrix.shape
        num_feats = len(windows) * 2
        rolling_features = np.zeros((N, T, num_feats), dtype=np.float32)

        for wi, w in enumerate(windows):
            for t in range(w, T):
                window_data = sales_matrix[:, t - w:t]
                # Guard: Ensure we don't have empty windows
                rolling_features[:, t, wi * 2] = np.mean(window_data, axis=1)
                rolling_features[:, t, wi * 2 + 1] = np.std(window_data, axis=1) + 1e-8

        return np.nan_to_num(rolling_features)

    def compute_price_features(
        self, 
        price_matrix: np.ndarray
    ) -> np.ndarray:
        N, T = price_matrix.shape
        price_feats = np.zeros((N, T, 3), dtype=np.float32)

        # 1. Raw price (normalized per item)
        price_mean = price_matrix.mean(axis=1, keepdims=True) + 1e-8
        price_feats[:, :, 0] = np.nan_to_num(price_matrix / price_mean)

        # 2. Price change (relative)
        denom = price_matrix[:, :-1] + 1e-8
        price_feats[:, 1:, 1] = np.nan_to_num((price_matrix[:, 1:] - price_matrix[:, :-1]) / denom)

        # 3. Price momentum
        w = self.feat_cfg.price_momentum_window
        for t in range(w, T):
            price_feats[:, t, 2] = np.mean(price_feats[:, t - w:t, 1], axis=1)

        return np.nan_to_num(price_feats)

    def encode_categories(
        self, 
        metadata: 'pd.DataFrame'
    ) -> Dict[str, np.ndarray]:
        encodings = {}
        for col in ['store_id', 'dept_id', 'cat_id', 'state_id', 'item_id']:
            if col in metadata.columns:
                categories = metadata[col].astype('category')
                encodings[col] = categories.cat.codes.values.astype(np.int64)
                encodings[f'{col}_vocab_size'] = len(categories.cat.categories)
        return encodings

    def build_stream_tensors(
        self, 
        dataset: Dict,
        start_day: int, 
        end_day: int,
        device: torch.device = torch.device('cpu'),
        item_features: 'np.ndarray | None' = None,
        precomputed: 'Dict | None' = None,
    ) -> Dict[str, torch.Tensor]:
        sales = dataset['sales_matrix']
        prices = dataset['price_matrix']
        cal_feats = dataset['calendar_features']
        metadata = dataset['metadata']
        N, T_total = sales.shape

        # ── Use precomputed features if available, otherwise compute ──
        if precomputed is not None:
            lag_feats = precomputed['lag_feats']
            rolling_feats = precomputed['rolling_feats']
            price_feats = precomputed['price_feats']
        else:
            lag_feats = self.compute_lag_features(sales, self.feat_cfg.lags)
            rolling_feats = self.compute_rolling_features(sales, self.feat_cfg.rolling_windows)
            price_feats = self.compute_price_features(prices)

        # ── Slice to the requested window ──
        # Clamp start_day to valid range
        start_day = max(0, start_day)
        end_day = min(end_day, T_total)
        
        window_sales = sales[:, start_day:end_day]
        window_lags = lag_feats[:, start_day:end_day, :]
        window_rolling = rolling_feats[:, start_day:end_day, :]
        window_prices = price_feats[:, start_day:end_day, :]
        window_cal = cal_feats[start_day:end_day, :]

        W = end_day - start_day
        window_cal_expanded = np.tile(window_cal[np.newaxis, :, :], (N, 1, 1))

        # ── Ensure all log inputs are >= 0 to avoid NaNs ──
        demand_feat = np.log1p(np.maximum(window_sales, 0))[:, :, np.newaxis]

        # ── Build feature list ──
        feature_arrays = [
            demand_feat,                             # 1: log demand
            np.log1p(np.maximum(window_lags, 0)),    # L: log lag features
            np.log1p(np.maximum(window_rolling, 0)), # R: log rolling features
            window_prices,                           # 3: price features
            window_cal_expanded,                     # 8: calendar features
        ]
        
        # ── Inject per-item static features (broadcast across time) ──
        if item_features is not None:
            # item_features: (N, F_static) → broadcast to (N, W, F_static)
            item_feat_expanded = np.tile(item_features[:, np.newaxis, :], (1, W, 1))
            feature_arrays.append(item_feat_expanded)

        # ── Concatenate features ──
        node_features = np.concatenate(feature_arrays, axis=-1)

        # ── NUMERICAL GUARD: Replace NaNs and Infs ──
        node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ── FIX: Per-item temporal standardization ──
        # Standardize each item's features across TIME only (axis=1),
        # preserving cross-item magnitude differences.
        # Global standardization (axis=(0,1)) erased the fact that popular items
        # sell 100x more than sparse items, making the GNN unable to distinguish them.
        feat_mean = node_features.mean(axis=1, keepdims=True)  # (N, 1, F)
        feat_std = node_features.std(axis=1, keepdims=True) + 1e-8  # (N, 1, F)
        node_features = (node_features - feat_mean) / feat_std
        
        # Clip extreme standardized values (beyond 5 sigma)
        node_features = np.clip(node_features, -5.0, 5.0)

        # ── Targets ──
        target_start = end_day
        target_end = min(end_day + self.data_cfg.horizon, T_total)
        if target_end > target_start:
            targets = np.nan_to_num(sales[:, target_start:target_end], nan=0.0)
        else:
            targets = np.zeros((N, self.data_cfg.horizon), dtype=np.float32)

        if targets.shape[1] < self.data_cfg.horizon:
            pad_width = self.data_cfg.horizon - targets.shape[1]
            targets = np.pad(targets, ((0, 0), (0, pad_width)), mode='constant')

        cat_ids = self.encode_categories(metadata)

        # ── Convert to tensors ──
        result = {
            'node_features': torch.tensor(node_features, dtype=torch.float32).to(device),
            'targets': torch.tensor(targets, dtype=torch.float32).to(device),
            'sales_history': torch.tensor(np.nan_to_num(sales[:, :end_day]), dtype=torch.float32).to(device),
            'category_ids': {
                k: torch.tensor(v, dtype=torch.long).to(device)
                for k, v in cat_ids.items()
                if not k.endswith('_vocab_size')
            },
            'category_vocab_sizes': {
                k: v for k, v in cat_ids.items()
                if k.endswith('_vocab_size')
            },
            'num_features': node_features.shape[-1],
        }

        return result

    def compute_dow_baseline(self, sales_matrix, end_day, horizon=28, num_weeks=8):
        """
        Day-of-week seasonal baseline — the strongest simple predictor in retail.
        
        For each item and each horizon day, computes the mean sales on the same 
        day-of-week from the last num_weeks weeks. This captures weekly seasonality
        which is the dominant pattern in Walmart sales.
        
        M5 Day 0 (d_1) = Sat Jan 29, 2011 → dow = (day_index + 5) % 7
        (Mon=0, Tue=1, ..., Sat=5, Sun=6)
        """
        N = sales_matrix.shape[0]
        baselines = np.zeros((N, horizon), dtype=np.float32)
        
        for h in range(horizon):
            target_day = end_day + h
            same_dow_sales = []
            for w in range(1, num_weeks + 1):
                past_day = target_day - 7 * w
                if 0 <= past_day < end_day:
                    same_dow_sales.append(sales_matrix[:, past_day])
            if same_dow_sales:
                baselines[:, h] = np.median(same_dow_sales, axis=0)
        
        return baselines

    def compute_item_features(self, sales_matrix, end_day):
        """
        Per-item static features that give the model explicit item context.
        Returns (N, 12) array: [log_mean, log_std, log_max, zero_frac, trend, dow_means×7]
        """
        train_sales = sales_matrix[:, :end_day].astype(np.float32)
        N = train_sales.shape[0]
        
        # Basic item stats
        item_mean = train_sales.mean(axis=1)
        item_std = train_sales.std(axis=1)
        item_max = train_sales.max(axis=1)
        item_zero_frac = (train_sales == 0).mean(axis=1)
        
        # Recent trend (vectorized slope of last 28 days)
        recent = train_sales[:, -28:]
        x = np.arange(28, dtype=np.float32)
        x_mean = x.mean()
        x_centered = x - x_mean
        recent_mean = recent.mean(axis=1, keepdims=True)
        recent_centered = recent - recent_mean
        slopes = (recent_centered * x_centered).sum(axis=1) / ((x_centered ** 2).sum() + 1e-8)
        
        # Per day-of-week means (captures weekly pattern)
        dow_means = np.zeros((N, 7), dtype=np.float32)
        for dow in range(7):
            dow_days = [d for d in range(max(0, end_day - 56), end_day) if (d + 5) % 7 == dow]
            if dow_days:
                dow_means[:, dow] = train_sales[:, dow_days].mean(axis=1)
        
        features = np.column_stack([
            np.log1p(item_mean),
            np.log1p(item_std),
            np.log1p(item_max),
            item_zero_frac,
            slopes,
            dow_means,
        ]).astype(np.float32)
        
        # Standardize
        feat_mean = features.mean(axis=0, keepdims=True)
        feat_std = features.std(axis=0, keepdims=True) + 1e-8
        features = np.clip((features - feat_mean) / feat_std, -5, 5)
        
        return features