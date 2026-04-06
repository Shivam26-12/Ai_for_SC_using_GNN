"""
Chaos Engineering Engine — Orchestrates perturbation experiments
with full Hawkes Process parameter sweep support.

Experiment Signature:
    E_k = (perturbation_type, severity, seed, μ, α, β)

When Hawkes is disabled (use_hawkes=False), reduces to:
    E_k = (perturbation_type, severity, seed)
"""
import torch
import numpy as np
import os
from itertools import product
from typing import Dict, List, Optional, Tuple
from .perturbations import (
    Perturbation, DemandShock, SupplyDisruption, PriceVolatility,
    CalendarShift, GraphCorruption, AdversarialAttack,
)
from .hawkes_process import HawkesProcess, HawkesParams


class ChaosEngine:
    """
    Orchestrates chaos engineering experiments with optional Hawkes Process.

    Usage:
        engine = ChaosEngine(config)
        results = engine.run_all(model, features, edge_index, edge_type, targets, ...)
    """

    def __init__(
        self,
        num_trials: int = 5,
        seed: int = 42,
        use_hawkes: bool = True,
        hawkes_mu_values: List[float] = None,
        hawkes_alpha_values: List[float] = None,
        hawkes_beta_values: List[float] = None,
        traces_dir: str = None,
    ):
        self.num_trials = num_trials
        self.seed = seed
        self.use_hawkes = use_hawkes

        # Hawkes parameter grid (defaults for adversarial regime search)
        self.hawkes_mu_values = hawkes_mu_values or [0.1]
        self.hawkes_alpha_values = hawkes_alpha_values or [0.6]
        self.hawkes_beta_values = hawkes_beta_values or [1.0]

        # Directory for saving intensity traces
        self.traces_dir = traces_dir
        if traces_dir:
            os.makedirs(traces_dir, exist_ok=True)

    def _create_hawkes(
        self, mu: float, alpha: float, beta: float, seed: int
    ) -> Optional[HawkesProcess]:
        """Create a HawkesProcess instance, or None if disabled."""
        if not self.use_hawkes or alpha == 0.0:
            return None
        params = HawkesParams(mu=mu, alpha=alpha, beta=beta)
        return HawkesProcess(params=params, seed=seed)

    def _build_perturbations(
        self, hawkes: Optional[HawkesProcess] = None
    ) -> Dict[str, Perturbation]:
        """Build all perturbation instances with optional Hawkes."""
        return {
            'demand_shock_spike': DemandShock(
                severity=0.5, shock_type='spike', seed=self.seed, hawkes=hawkes
            ),
            'demand_shock_crash': DemandShock(
                severity=0.5, shock_type='crash', seed=self.seed + 1, hawkes=hawkes
            ),
            'supply_disruption': SupplyDisruption(
                severity=0.5, seed=self.seed + 2, hawkes=hawkes
            ),
            'price_volatility': PriceVolatility(
                severity=0.3, seed=self.seed + 3, hawkes=hawkes
            ),
            'calendar_shift': CalendarShift(
                severity=0.5, max_shift=3, seed=self.seed + 4, hawkes=hawkes
            ),
            'graph_corruption_10': GraphCorruption(
                severity=0.5, drop_ratio=0.1, seed=self.seed + 5, hawkes=hawkes
            ),
            'graph_corruption_30': GraphCorruption(
                severity=1.0, drop_ratio=0.3, seed=self.seed + 6, hawkes=hawkes
            ),
            'adversarial_fgsm': AdversarialAttack(
                epsilon=0.01, method='fgsm', seed=self.seed + 7, hawkes=hawkes
            ),
            'adversarial_pgd': AdversarialAttack(
                epsilon=0.01, num_steps=5, method='pgd', seed=self.seed + 8, hawkes=hawkes
            ),
        }

    def run_single(
        self,
        name: str,
        perturbation: Perturbation,
        model: torch.nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        clean_predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[torch.nn.Module] = None,
        **model_kwargs,
    ) -> Dict[str, float]:
        """Run a single perturbation and measure impact."""
        model.eval()

        if isinstance(perturbation, AdversarialAttack) and targets is not None and loss_fn is not None:
            perturbed_features, perturbed_edges, perturbed_types = perturbation.apply(
                node_features, edge_index, edge_type,
                model=model, targets=targets, loss_fn=loss_fn,
                **model_kwargs,
            )
        else:
            with torch.no_grad():
                perturbed_features, perturbed_edges, perturbed_types = perturbation.apply(
                    node_features, edge_index, edge_type,
                    **model_kwargs,
                )

        with torch.no_grad():
            chaos_predictions = model(
                perturbed_features, perturbed_edges, perturbed_types,
                model_kwargs.get('category_ids', {}),
                model_kwargs.get('dept_ids'),
                model_kwargs.get('historical_mean'),
            )

        # Compute stability metrics
        pred_diff = torch.abs(clean_predictions - chaos_predictions)
        mean_deviation = pred_diff.mean().item()
        max_deviation = pred_diff.max().item()
        relative_change = (pred_diff / (torch.abs(clean_predictions) + 1e-8)).mean().item()

        stability = 1.0 - min(relative_change, 1.0)

        result = {
            'perturbation': name,
            'mean_deviation': mean_deviation,
            'max_deviation': max_deviation,
            'relative_change': relative_change,
            'stability_score': stability,
        }

        # Add Hawkes stats if available
        if perturbation.hawkes is not None:
            hawkes_stats = perturbation.hawkes.get_summary_stats()
            result['hawkes_n_events'] = hawkes_stats['n_events']
            result['hawkes_lambda_mean'] = hawkes_stats['lambda_mean']
            result['hawkes_lambda_max'] = hawkes_stats['lambda_max']
            result['hawkes_branching_ratio'] = hawkes_stats['branching_ratio']
            result['hawkes_mu'] = perturbation.hawkes.params.mu
            result['hawkes_alpha'] = perturbation.hawkes.params.alpha
            result['hawkes_beta'] = perturbation.hawkes.params.beta

        return result

    def run_all(
        self,
        model: torch.nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[torch.nn.Module] = None,
        **model_kwargs,
    ) -> List[Dict[str, float]]:
        """Run all perturbations and collect results."""
        model.eval()

        with torch.no_grad():
            clean_predictions = model(
                node_features, edge_index, edge_type,
                model_kwargs.get('category_ids', {}),
                model_kwargs.get('dept_ids'),
                model_kwargs.get('historical_mean'),
            )

        results = []

        # Build Hawkes parameter grid
        if self.use_hawkes:
            hawkes_configs = list(product(
                self.hawkes_mu_values,
                self.hawkes_alpha_values,
                self.hawkes_beta_values,
            ))
            print(f"\n   🔬 Hawkes parameter grid: {len(hawkes_configs)} configurations")
        else:
            hawkes_configs = [(0.0, 0.0, 1.0)]  # α=0 → Bernoulli fallback

        for hi, (mu, alpha, beta) in enumerate(hawkes_configs):
            if self.use_hawkes and alpha > 0:
                print(f"\n   ═══ Hawkes config {hi+1}/{len(hawkes_configs)}: "
                      f"μ={mu}, α={alpha}, β={beta} (ratio={alpha/beta:.2f}) ═══")

            hawkes = self._create_hawkes(mu, alpha, beta, self.seed)
            perturbations = self._build_perturbations(hawkes)

            for name, perturbation in perturbations.items():
                label = name
                if self.use_hawkes and alpha > 0:
                    label = f"{name} [H:μ={mu},α={alpha},β={beta}]"

                print(f"   🔥 {label}...")

                # Reset Hawkes for each perturbation
                if perturbation.hawkes is not None:
                    perturbation.hawkes.reset()

                result = self.run_single(
                    label, perturbation, model,
                    node_features, edge_index, edge_type,
                    clean_predictions, targets, loss_fn,
                    **model_kwargs,
                )
                results.append(result)

                # Save intensity trace
                if perturbation.hawkes is not None and self.traces_dir:
                    trace_name = f"trace_{name}_mu{mu}_a{alpha}_b{beta}.npz"
                    trace_path = os.path.join(self.traces_dir, trace_name)
                    try:
                        perturbation.hawkes.save_trace(trace_path)
                    except Exception:
                        pass

                print(f"      Stability: {result['stability_score']:.4f} | "
                      f"Mean Δ: {result['mean_deviation']:.4f}"
                      + (f" | Events: {result.get('hawkes_n_events', 'N/A')}"
                         if self.use_hawkes and alpha > 0 else ""))

        return results
