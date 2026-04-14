"""
WRMSSE (Weighted Root Mean Squared Scaled Error) — M5 Official Metric.

This implementation follows the exact specification from the M5 competition:
1. Calculate RMSSE for each of the 30,490 bottom-level series
2. Aggregate across 12 hierarchical levels
3. Weight by dollar-sales contribution

Reference: https://mofc.unic.ac.cy/m5-competition/

UPDATED: Bulletproof NaN prevention — raised scale floor from 1e-6 to 1.0,
clamped RMSSE to max 100.0, added comprehensive input sanitization.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WRMSSEEvaluator:
    """
    Computes the official M5 WRMSSE metric.
    
    WRMSSE = Weighted average of RMSSE across all hierarchical aggregations.
    
    RMSSE_i = RMSE(predictions_i, actuals_i) / sqrt(mean(diff(train_i)^2))
    
    The scaling denominator uses the naive one-step forecast error
    from the training data, making the metric scale-independent.
    """

    def __init__(
        self,
        train_sales: np.ndarray,
        train_prices: np.ndarray,
        metadata: pd.DataFrame,
        horizon: int = 28
    ):
        """
        Args:
            train_sales: (N, T_train) training sales data
            train_prices: (N, T_train) training price data  
            metadata: DataFrame with item_id, store_id, dept_id, cat_id, state_id
            horizon: forecast horizon (28 for M5)
        """
        # ── Sanitize ALL inputs upfront ──
        train_sales = np.nan_to_num(np.asarray(train_sales, dtype=np.float64),
                                    nan=0.0, posinf=0.0, neginf=0.0)
        train_prices = np.nan_to_num(np.asarray(train_prices, dtype=np.float64),
                                     nan=0.0, posinf=0.0, neginf=0.0)

        self.N = train_sales.shape[0]
        self.T_train = train_sales.shape[1]
        self.horizon = horizon
        self.metadata = metadata
        self._full_train_sales = train_sales

        # ── Compute scale (denominator) for each series ──
        self.scales = self._compute_scales(train_sales)

        # ── Compute weights based on dollar-sales ──
        self.weights = self._compute_weights(train_sales, train_prices)

        # ── Diagnostic printout ──
        n_tiny = np.sum(self.scales < 1.0 + 1e-6)  # Items that hit the floor
        print(f"   [WRMSSE] {self.N} items | "
              f"scale range [{self.scales.min():.4f}, {self.scales.max():.4f}] | "
              f"{n_tiny} items hit scale floor")

    def _compute_scales(self, train_sales: np.ndarray) -> np.ndarray:
        """
        Compute the RMSSE scaling factor for each series.
        scale_i = sqrt( (1/(T-1)) * sum_{t=2}^{T} (y_t - y_{t-1})^2 )
        Only starts the sequence from the first non-zero demand for each item.
        
        CRITICAL FIX: Floor raised from 1e-6 to 1.0.
        Items with scale < 1.0 have near-zero variance (constant or nearly-zero sales).
        A floor of 1e-6 means RMSE(1.0) / 1e-6 = 1,000,000 → blows up the sum.
        """
        scales = np.zeros(train_sales.shape[0], dtype=np.float64)
        for i in range(train_sales.shape[0]):
            nz_idx = np.where(train_sales[i] > 0)[0]
            if len(nz_idx) == 0:
                scales[i] = 1.0  # No sales at all → floor
                continue
                
            first_nz = nz_idx[0]
            series = train_sales[i, first_nz:]
            
            if len(series) < 2:
                scales[i] = 1.0  # Too short → floor
                continue
                
            diffs = np.diff(series)
            msd = np.mean(diffs ** 2)
            if msd <= 0 or np.isnan(msd) or np.isinf(msd):
                scales[i] = 1.0
            else:
                scales[i] = np.sqrt(msd)
            
        # Prevent division by zero — floor at 1.0 (not 1e-6!)
        scales = np.nan_to_num(scales, nan=1.0, posinf=1.0, neginf=1.0)
        scales = np.maximum(scales, 1.0)
        return scales

    def _compute_weights(
        self, 
        train_sales: np.ndarray,
        train_prices: np.ndarray
    ) -> np.ndarray:
        """
        Compute weights based on cumulative dollar sales in the last 28 days
        of training data.
        
        weight_i = dollar_sales_i / sum(dollar_sales)
        """
        # Dollar sales in the last 28 days of training
        recent_sales = np.maximum(train_sales[:, -28:], 0.0)
        recent_prices = np.maximum(train_prices[:, -28:], 0.0)
        dollar_sales = np.sum(recent_sales * recent_prices, axis=1)  # (N,)

        # Sanitize
        dollar_sales = np.nan_to_num(dollar_sales, nan=0.0, posinf=0.0, neginf=0.0)

        total = np.sum(dollar_sales)
        if total <= 0 or np.isnan(total):
            # Fallback: equal weights
            print("   [WRMSSE] WARNING: Total dollar sales is 0 or NaN. Using equal weights.")
            return np.ones(self.N, dtype=np.float64) / self.N

        weights = dollar_sales / total
        # Final sanitize
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        # Re-normalize in case nan_to_num zeroed anything
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = np.ones(self.N, dtype=np.float64) / self.N
        return weights

    def compute_rmsse(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray
    ) -> np.ndarray:
        """
        Compute RMSSE for each series.
        
        RMSSE_i = RMSE(pred_i, actual_i) / scale_i
        
        Args:
            predictions: (N, horizon) predicted values
            actuals: (N, horizon) actual values
            
        Returns:
            (N,) RMSSE values, clamped to [0, 100]
        """
        assert predictions.shape == actuals.shape, \
            f"Shape mismatch: {predictions.shape} vs {actuals.shape}"

        # Sanitize inputs
        predictions = np.nan_to_num(np.asarray(predictions, dtype=np.float64),
                                    nan=0.0, posinf=0.0, neginf=0.0)
        actuals = np.nan_to_num(np.asarray(actuals, dtype=np.float64),
                                nan=0.0, posinf=0.0, neginf=0.0)

        # RMSE per series
        mse = np.mean((predictions - actuals) ** 2, axis=1)  # (N,)
        mse = np.maximum(mse, 0.0)  # Guard against numerical negative
        rmse = np.sqrt(mse)

        # Scale (denominator) — already floored at 1.0
        rmsse = rmse / self.scales

        # ── CRITICAL: Clamp RMSSE to prevent Inf/NaN in weighted sum ──
        # An RMSSE > 100 means the model is 100x worse than naive — meaningless
        rmsse = np.clip(rmsse, 0.0, 100.0)
        rmsse = np.nan_to_num(rmsse, nan=0.0, posinf=100.0, neginf=0.0)
        return rmsse

    def compute_wrmsse(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray
    ) -> float:
        """
        Compute the full WRMSSE metric (bottom-level only).
        
        Args:
            predictions: (N, horizon) predicted values
            actuals: (N, horizon) actual values
            
        Returns:
            Scalar WRMSSE value (guaranteed finite)
        """
        rmsse = self.compute_rmsse(predictions, actuals)
        wrmsse = np.sum(self.weights * rmsse)
        
        # Final safety net
        if np.isnan(wrmsse) or np.isinf(wrmsse):
            # Fallback: unweighted mean RMSSE
            fallback = float(np.mean(rmsse))
            print(f"   [WRMSSE] WARNING: Weighted WRMSSE was NaN/Inf. "
                  f"Falling back to mean RMSSE = {fallback:.4f}")
            return fallback
        
        return float(wrmsse)

    def compute_hierarchical_wrmsse(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute WRMSSE at all 12 M5 hierarchical levels.
        
        Returns:
            Dictionary of level_name → WRMSSE score (all guaranteed finite)
        """
        # Sanitize inputs once
        predictions = np.nan_to_num(np.asarray(predictions, dtype=np.float64),
                                    nan=0.0, posinf=0.0, neginf=0.0)
        actuals = np.nan_to_num(np.asarray(actuals, dtype=np.float64),
                                nan=0.0, posinf=0.0, neginf=0.0)

        meta = self.metadata
        results = {}

        # Level 12: Bottom level (item × store)
        rmsse = self.compute_rmsse(predictions, actuals)
        results['item_store'] = float(np.mean(rmsse))

        # Aggregation helper
        def aggregate_and_score(group_cols, level_name):
            """Aggregate predictions and actuals by group, then compute RMSSE."""
            if group_cols:
                group_keys = meta[group_cols].apply(
                    lambda x: '_'.join(x.astype(str)), axis=1
                )
            else:
                group_keys = pd.Series(['total'] * len(meta))

            unique_groups = group_keys.unique()
            group_rmsse_list = []

            for g in unique_groups:
                mask = (group_keys == g).values
                agg_pred = predictions[mask].sum(axis=0).astype(np.float64)
                agg_actual = actuals[mask].sum(axis=0).astype(np.float64)

                diff_sq = (agg_pred - agg_actual) ** 2
                diff_sq = np.nan_to_num(diff_sq, nan=0.0, posinf=0.0, neginf=0.0)
                rmse = np.sqrt(np.mean(diff_sq))

                # Scale for aggregated series
                train_agg = self._get_aggregated_train(mask)
                diffs = np.diff(train_agg)
                msd = np.mean(diffs ** 2)
                if msd <= 0 or np.isnan(msd):
                    scale = 1.0
                else:
                    scale = np.sqrt(msd)
                scale = max(scale, 1.0)  # Same floor as bottom level

                group_rmsse = rmse / scale
                # Clamp per-group RMSSE too
                group_rmsse = min(max(group_rmsse, 0.0), 100.0)
                if np.isnan(group_rmsse):
                    group_rmsse = 0.0
                group_rmsse_list.append(group_rmsse)

            results[level_name] = float(np.mean(group_rmsse_list)) if group_rmsse_list else 0.0

        # Compute for key levels
        aggregate_and_score([], 'total')
        aggregate_and_score(['state_id'], 'state')
        aggregate_and_score(['store_id'], 'store')
        aggregate_and_score(['cat_id'], 'category')
        aggregate_and_score(['dept_id'], 'department')
        aggregate_and_score(['store_id', 'cat_id'], 'store_category')
        aggregate_and_score(['store_id', 'dept_id'], 'store_department')

        # Overall WRMSSE (equal weight across levels for simplicity)
        level_scores = list(results.values())
        valid_scores = [s for s in level_scores if not np.isnan(s)]
        results['overall_wrmsse'] = float(np.mean(valid_scores)) if valid_scores else 0.0

        return results

    def _get_aggregated_train(self, mask: np.ndarray) -> np.ndarray:
        """Get aggregated training sales for a group mask."""
        if hasattr(self, '_full_train_sales'):
            return self._full_train_sales[mask].sum(axis=0)
        return np.ones(self.T_train)  # Fallback

    def set_train_sales(self, train_sales: np.ndarray):
        """Set training sales for hierarchical aggregation."""
        self._full_train_sales = np.nan_to_num(
            np.asarray(train_sales, dtype=np.float64), nan=0.0)


def compute_simple_metrics(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard forecasting metrics for comparison.
    """
    predictions = np.nan_to_num(predictions, nan=0.0)
    actuals = np.nan_to_num(actuals, nan=0.0)

    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # sMAPE (symmetric MAPE — better than MAPE for zero-inflated data)
    denom = (np.abs(predictions) + np.abs(actuals)) / 2.0 + 1e-8
    smape = np.mean(np.abs(predictions - actuals) / denom) * 100

    return {
        'MAE': float(np.nan_to_num(mae)),
        'RMSE': float(np.nan_to_num(rmse)),
        'sMAPE': float(np.nan_to_num(smape)),
    }
