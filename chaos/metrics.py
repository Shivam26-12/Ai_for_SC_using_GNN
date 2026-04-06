"""
Resilience Metrics for Chaos Engineering Evaluation.
Extended with Hawkes-specific robustness metrics.
"""
import numpy as np
from typing import Dict, List


class ResilienceMetrics:
    """Computes principled resilience metrics from chaos experiment results."""

    @staticmethod
    def prediction_stability(results: List[Dict]) -> float:
        """Average stability across all perturbation types."""
        scores = [r['stability_score'] for r in results]
        return float(np.mean(scores))

    @staticmethod
    def worst_case_stability(results: List[Dict]) -> float:
        """Minimum stability (worst perturbation)."""
        scores = [r['stability_score'] for r in results]
        return float(np.min(scores))

    @staticmethod
    def robustness_profile(results: List[Dict]) -> Dict[str, float]:
        """Categorized robustness scores."""
        categories = {
            'demand': ['demand_shock_spike', 'demand_shock_crash'],
            'supply': ['supply_disruption'],
            'economic': ['price_volatility'],
            'temporal': ['calendar_shift'],
            'structural': ['graph_corruption_10', 'graph_corruption_30'],
            'adversarial': ['adversarial_fgsm', 'adversarial_pgd'],
        }
        profile = {}
        for cat, names in categories.items():
            cat_results = [
                r for r in results
                if any(n in r['perturbation'] for n in names)
            ]
            if cat_results:
                profile[cat] = float(np.mean([r['stability_score'] for r in cat_results]))

        profile['overall'] = float(np.mean(list(profile.values()))) if profile else 0.0
        return profile

    @staticmethod
    def hawkes_robustness(results: List[Dict]) -> Dict[str, float]:
        """
        Hawkes-specific robustness analysis.

        R(μ, α, β) = stability_baseline / stability_hawkes

        Identifies the most adversarial Hawkes parameter regime.
        """
        hawkes_results = [r for r in results if r.get('hawkes_alpha', 0) > 0]
        non_hawkes = [r for r in results if r.get('hawkes_alpha', 0) == 0]

        if not hawkes_results:
            return {}

        baseline_stability = (
            float(np.mean([r['stability_score'] for r in non_hawkes]))
            if non_hawkes else 1.0
        )

        analysis = {
            'baseline_stability': baseline_stability,
            'hawkes_mean_stability': float(np.mean([
                r['stability_score'] for r in hawkes_results
            ])),
            'hawkes_worst_stability': float(np.min([
                r['stability_score'] for r in hawkes_results
            ])),
        }

        # Find most adversarial regime
        worst = min(hawkes_results, key=lambda r: r['stability_score'])
        analysis['worst_mu'] = worst.get('hawkes_mu', 0)
        analysis['worst_alpha'] = worst.get('hawkes_alpha', 0)
        analysis['worst_beta'] = worst.get('hawkes_beta', 1)
        analysis['worst_perturbation'] = worst['perturbation']
        analysis['worst_stability'] = worst['stability_score']

        # Robustness ratio
        if analysis['hawkes_worst_stability'] > 0:
            analysis['robustness_ratio'] = (
                baseline_stability / analysis['hawkes_worst_stability']
            )
        else:
            analysis['robustness_ratio'] = float('inf')

        # Average events triggered
        analysis['avg_cascade_events'] = float(np.mean([
            r.get('hawkes_n_events', 0) for r in hawkes_results
        ]))

        return analysis

    @staticmethod
    def summary_table(results: List[Dict]) -> str:
        """Pretty-print results table."""
        lines = []
        lines.append("=" * 80)
        lines.append(
            f"{'Perturbation':<40} {'Stability':>10} {'Mean Δ':>12} {'Rel Δ':>10}"
        )
        lines.append("=" * 80)
        for r in results:
            name = r['perturbation'][:38]
            lines.append(
                f"{name:<40} "
                f"{r['stability_score']:>10.4f} "
                f"{r['mean_deviation']:>12.4f} "
                f"{r['relative_change']:>10.4f}"
            )
        lines.append("=" * 80)

        # Robustness profile
        profile = ResilienceMetrics.robustness_profile(results)
        lines.append(f"\nOverall Resilience Score: {profile.get('overall', 0):.4f}")
        for cat, score in profile.items():
            if cat != 'overall':
                lines.append(f"  {cat:<20}: {score:.4f}")

        # Hawkes analysis
        hawkes_analysis = ResilienceMetrics.hawkes_robustness(results)
        if hawkes_analysis:
            lines.append(f"\n{'─'*40}")
            lines.append("Hawkes Cascade Analysis:")
            lines.append(f"  Baseline stability : {hawkes_analysis['baseline_stability']:.4f}")
            lines.append(f"  Hawkes mean stab.  : {hawkes_analysis['hawkes_mean_stability']:.4f}")
            lines.append(f"  Hawkes worst stab. : {hawkes_analysis['hawkes_worst_stability']:.4f}")
            lines.append(f"  Robustness ratio   : {hawkes_analysis['robustness_ratio']:.4f}")
            lines.append(f"  Avg cascade events : {hawkes_analysis['avg_cascade_events']:.1f}")
            lines.append(f"  Worst regime       : μ={hawkes_analysis['worst_mu']}, "
                         f"α={hawkes_analysis['worst_alpha']}, "
                         f"β={hawkes_analysis['worst_beta']}")

        return "\n".join(lines)
