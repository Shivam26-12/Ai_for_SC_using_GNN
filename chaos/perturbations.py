"""
Chaos Engineering — Perturbation Strategies with Hawkes Process Integration.

Implements 6 supply chain disruption scenarios, each upgraded with optional
Hawkes-driven self-exciting cascading failures:

1. Demand Shock: Sudden demand spike or crash (cascading with Hawkes)
2. Supply Disruption: Stockout, shipment delay (clustered with Hawkes)
3. Price Volatility: Competitor price war, inflation (intensity-scaled)
4. Calendar Shift: Holiday moved, unexpected event
5. Graph Corruption: Store closure, supply chain reconfiguration (cascading)
6. Adversarial Attack: Gradient-based perturbation (FGSM/PGD)

When hawkes=None, all perturbations behave identically to the original
Bernoulli-based implementations (full backward compatibility).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

from .hawkes_process import HawkesProcess, HawkesParams


class Perturbation(ABC):
    """Base class for all perturbations."""

    def __init__(self, severity: float = 0.5, seed: Optional[int] = None,
                 hawkes: Optional[HawkesProcess] = None):
        self.severity = severity
        self.rng = np.random.RandomState(seed)
        self.hawkes = hawkes

    @abstractmethod
    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply perturbation and return modified inputs.

        Returns:
            (node_features, edge_index, edge_type) — modified versions
        """
        pass

    def _get_hawkes_mask(self, n_rows: int, n_cols: int) -> np.ndarray:
        """Generate a Hawkes-driven failure mask if hawkes is set."""
        if self.hawkes is not None:
            return self.hawkes.simulate_2d(n_rows, n_cols)
        return None

    def _get_hawkes_scale(self, t: float) -> float:
        """Get intensity scale factor λ(t)/μ from Hawkes process."""
        if self.hawkes is not None:
            return self.hawkes.intensity_scale_factor(t)
        return 1.0

    def __repr__(self):
        hawkes_str = f", hawkes={self.hawkes.params}" if self.hawkes else ""
        return f"{self.__class__.__name__}(severity={self.severity}{hawkes_str})"


class DemandShock(Perturbation):
    """
    Simulates sudden demand spikes or crashes with optional Hawkes cascading.

    With Hawkes: Each shock event excites subsequent shocks. A demand crash
    at time t increases the probability of crashes at t+1, t+2, ...
    mimicking panic-buying contagion or cascading supply failures.

    Without Hawkes: Standard Bernoulli random shocks (original behavior).
    """

    def __init__(
        self,
        severity: float = 0.5,
        window_size: int = 14,
        item_fraction: float = 0.2,
        shock_type: str = 'mixed',
        seed: Optional[int] = None,
        hawkes: Optional[HawkesProcess] = None,
    ):
        super().__init__(severity, seed, hawkes)
        self.window_size = window_size
        self.item_fraction = item_fraction
        self.shock_type = shock_type

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, C = node_features.shape
        features = node_features.clone()

        if self.hawkes is not None:
            # Hawkes-driven: iterate over items, cascading shocks
            shocked_items = []
            n_base = max(1, int(N * self.item_fraction))

            # First wave: guaranteed shocks
            first_wave = self.rng.choice(N, n_base, replace=False)
            shocked_items.extend(first_wave.tolist())
            for idx in first_wave:
                self.hawkes.record_event(float(idx))

            # Cascading shocks driven by Hawkes
            for i in range(N):
                if i in shocked_items:
                    continue
                p = self.hawkes.failure_probability(float(i))
                if self.rng.uniform() < p:
                    shocked_items.append(i)
                    self.hawkes.record_event(float(i))
        else:
            # Original Bernoulli
            num_shocked = max(1, int(N * self.item_fraction))
            shocked_items = self.rng.choice(N, num_shocked, replace=False).tolist()

        # Select random window
        max_start = max(0, T - self.window_size)
        window_start = self.rng.randint(0, max_start + 1)
        window_end = min(window_start + self.window_size, T)

        # Apply shocks with intensity scaling
        for item in shocked_items:
            # Scale shock factor by Hawkes intensity
            scale = self._get_hawkes_scale(float(item))

            if self.shock_type == 'spike':
                factor = 1.0 + self.severity * scale * (5.0 + self.rng.exponential(3.0))
            elif self.shock_type == 'crash':
                factor = max(0.01, 1.0 - self.severity * scale * self.rng.uniform(0.5, 1.0))
            else:  # mixed
                if self.rng.random() > 0.5:
                    factor = 1.0 + self.severity * scale * (5.0 + self.rng.exponential(3.0))
                else:
                    factor = max(0.01, 1.0 - self.severity * scale * self.rng.uniform(0.5, 1.0))

            # Apply to demand feature (channel 0 = log1p(demand))
            features[item, window_start:window_end, 0] = (
                torch.expm1(features[item, window_start:window_end, 0]) * factor
            )
            features[item, window_start:window_end, 0] = torch.log1p(
                F.relu(features[item, window_start:window_end, 0])
            )

        return features, edge_index, edge_type


class SupplyDisruption(Perturbation):
    """
    Simulates stockouts and supply chain disruptions with Hawkes clustering.

    With Hawkes: Disruption events cluster in time — one stockout excites
    subsequent stockouts, modeling cascading warehouse/supplier failures.

    Without Hawkes: Standard random stockouts (original behavior).
    """

    def __init__(
        self,
        severity: float = 0.5,
        window_size: int = 14,
        item_fraction: float = 0.1,
        seed: Optional[int] = None,
        hawkes: Optional[HawkesProcess] = None,
    ):
        super().__init__(severity, seed, hawkes)
        self.window_size = window_size
        self.item_fraction = item_fraction

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, C = node_features.shape
        features = node_features.clone()

        if self.hawkes is not None:
            # Hawkes-driven: simulate 2D mask (items × time)
            mask = self.hawkes.simulate_2d(N, T)
            # Apply mask to demand channel
            mask_t = torch.tensor(mask, dtype=features.dtype, device=features.device)
            features[:, :, 0] = features[:, :, 0] * mask_t
        else:
            # Original Bernoulli
            num_disrupted = max(1, int(N * self.item_fraction * self.severity))
            disrupted_items = self.rng.choice(N, num_disrupted, replace=False)
            win = int(self.window_size * self.severity)
            for item in disrupted_items:
                start = self.rng.randint(0, max(1, T - win))
                end = min(start + win, T)
                features[item, start:end, 0] = 0.0

        return features, edge_index, edge_type


class PriceVolatility(Perturbation):
    """
    Simulates price instability with intensity-scaled noise.

    With Hawkes: σ_eff(t) = σ_base · (λ(t) / μ)
    Noise magnitude increases during cascade bursts.

    Without Hawkes: Standard heavy-tailed noise (original behavior).
    """

    def __init__(
        self,
        severity: float = 0.3,
        seed: Optional[int] = None,
        hawkes: Optional[HawkesProcess] = None,
    ):
        super().__init__(severity, seed, hawkes)

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        price_channel_idx: int = -6,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, C = node_features.shape
        features = node_features.clone()

        # Generate heavy-tailed noise (Student-t with df=3)
        base_noise = torch.from_numpy(
            self.rng.standard_t(3, size=(N, T))
        ).float().to(features.device)

        if self.hawkes is not None:
            # Intensity-scaled noise: compute scale factors per timestep
            scale_factors = np.ones(T)
            self.hawkes.intensity_trace = []
            for t in range(T):
                t_float = float(t)
                lam = self.hawkes.compute_intensity(t_float)
                self.hawkes.intensity_trace.append(lam)
                sf = lam / self.hawkes.params.mu if self.hawkes.params.mu > 0 else 1.0
                scale_factors[t] = sf

                # Noise event can trigger more noise
                p = 1.0 - np.exp(-lam * self.hawkes.dt)
                if self.rng.uniform() < p:
                    self.hawkes.record_event(t_float)

            sf_tensor = torch.tensor(
                scale_factors, dtype=torch.float32, device=features.device
            ).unsqueeze(0)  # (1, T)
            noise = 1.0 + self.severity * base_noise * 0.3 * sf_tensor
        else:
            noise = 1.0 + self.severity * base_noise * 0.3

        # Apply to price-related channels
        price_start = 15
        price_end = min(price_start + 3, C)

        if price_start < C:
            for ch in range(price_start, price_end):
                features[:, :, ch] = features[:, :, ch] * noise

        return features, edge_index, edge_type


class CalendarShift(Perturbation):
    """
    Simulates calendar anomalies.

    With Hawkes: Shift magnitude scales with intensity.
    Without Hawkes: Standard circular shift (original behavior).
    """

    def __init__(
        self,
        severity: float = 0.5,
        max_shift: int = 3,
        seed: Optional[int] = None,
        hawkes: Optional[HawkesProcess] = None,
    ):
        super().__init__(severity, seed, hawkes)
        self.max_shift = max_shift

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, C = node_features.shape
        features = node_features.clone()

        cal_start = C - 8

        if self.hawkes is not None:
            # Scale shift by Hawkes intensity
            scale = self._get_hawkes_scale(0.0)
            max_shift_scaled = int(self.max_shift * min(scale, 3.0))
            shift = self.rng.randint(-max_shift_scaled, max_shift_scaled + 1)
        else:
            shift = self.rng.randint(-self.max_shift, self.max_shift + 1)

        if shift != 0 and cal_start < C:
            features[:, :, cal_start:] = torch.roll(
                features[:, :, cal_start:], shift, dims=1
            )

        return features, edge_index, edge_type


class GraphCorruption(Perturbation):
    """
    Simulates supply chain graph disruptions with Hawkes cascading.

    With Hawkes: Edge drop fraction scales with cascading intensity.
    First store closure excites additional closures.

    Without Hawkes: Fixed random edge drops (original behavior).
    """

    def __init__(
        self,
        severity: float = 0.2,
        drop_ratio: float = 0.2,
        add_noise_edges: bool = False,
        seed: Optional[int] = None,
        hawkes: Optional[HawkesProcess] = None,
    ):
        super().__init__(severity, seed, hawkes)
        self.drop_ratio = drop_ratio
        self.add_noise_edges = add_noise_edges

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E = edge_index.size(1)
        N = node_features.size(0)

        if self.hawkes is not None:
            # Hawkes-scaled: drop ratio increases with intensity
            scale = self._get_hawkes_scale(0.0)
            effective_ratio = min(self.drop_ratio * self.severity * scale, 0.8)
            self.hawkes.record_event(0.0)
        else:
            effective_ratio = self.drop_ratio * self.severity

        num_drop = int(E * effective_ratio)
        keep_mask = torch.ones(E, dtype=torch.bool, device=edge_index.device)

        if num_drop > 0 and num_drop < E:
            drop_indices = self.rng.choice(E, num_drop, replace=False)
            keep_mask[drop_indices] = False

        new_edge_index = edge_index[:, keep_mask]
        new_edge_type = edge_type[keep_mask]

        # Optionally add noise edges
        if self.add_noise_edges:
            num_noise = int(num_drop * 0.5)
            noise_src = torch.randint(0, N, (num_noise,), device=edge_index.device)
            noise_dst = torch.randint(0, N, (num_noise,), device=edge_index.device)
            noise_edges = torch.stack([noise_src, noise_dst])
            noise_types = torch.zeros(num_noise, dtype=torch.long, device=edge_index.device)

            new_edge_index = torch.cat([new_edge_index, noise_edges], dim=1)
            new_edge_type = torch.cat([new_edge_type, noise_types])

        return node_features, new_edge_index, new_edge_type


class AdversarialAttack(Perturbation):
    """
    Gradient-based adversarial perturbation (FGSM / PGD).

    FGSM: x_adv = x + ε · sign(∇_x L(model(x), y))
    PGD: Iterative FGSM with projection onto ε-ball.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        num_steps: int = 5,
        step_size: Optional[float] = None,
        method: str = 'pgd',
        seed: Optional[int] = None,
        hawkes: Optional[HawkesProcess] = None,
    ):
        super().__init__(epsilon, seed, hawkes)
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size or (epsilon / max(num_steps, 1) * 2.0)
        self.method = method

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        model: Optional[nn.Module] = None,
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if model is None or targets is None or loss_fn is None:
            noise = torch.empty_like(node_features).uniform_(
                -self.epsilon, self.epsilon
            )
            return node_features + noise, edge_index, edge_type

        was_training = model.training
        model.eval()

        # Scale epsilon by Hawkes intensity if active
        effective_eps = self.epsilon
        if self.hawkes is not None:
            scale = self._get_hawkes_scale(0.0)
            effective_eps = self.epsilon * min(scale, 3.0)

        if self.method == 'fgsm':
            adv_features = self._fgsm(
                model, node_features, edge_index, edge_type,
                targets, loss_fn, effective_eps, **kwargs
            )
        else:
            adv_features = self._pgd(
                model, node_features, edge_index, edge_type,
                targets, loss_fn, effective_eps, **kwargs
            )

        if was_training:
            model.train()

        return adv_features, edge_index, edge_type

    def _fgsm(
        self, model, features, edge_index, edge_type,
        targets, loss_fn, eps, **kwargs
    ):
        features = features.clone().detach().requires_grad_(True)

        predictions = model(
            features, edge_index, edge_type,
            kwargs.get('category_ids', {}),
            kwargs.get('dept_ids'),
            kwargs.get('historical_mean'),
        )
        loss = loss_fn(predictions, targets)
        loss.backward()

        perturbation = eps * features.grad.sign()
        adv_features = features.detach() + perturbation

        return adv_features

    def _pgd(
        self, model, features, edge_index, edge_type,
        targets, loss_fn, eps, **kwargs
    ):
        adv_features = features.clone().detach()
        adv_features = adv_features + torch.empty_like(adv_features).uniform_(
            -eps, eps
        )

        step_size = eps / max(self.num_steps, 1) * 2.0

        for _ in range(self.num_steps):
            adv_features = adv_features.clone().detach().requires_grad_(True)

            predictions = model(
                adv_features, edge_index, edge_type,
                kwargs.get('category_ids', {}),
                kwargs.get('dept_ids'),
                kwargs.get('historical_mean'),
            )
            loss = loss_fn(predictions, targets)
            loss.backward()

            grad = adv_features.grad
            adv_features = adv_features.detach() + step_size * grad.sign()

            perturbation = adv_features - features
            perturbation = torch.clamp(perturbation, -eps, eps)
            adv_features = features + perturbation

        return adv_features.detach()
