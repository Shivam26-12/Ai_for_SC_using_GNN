"""
Hawkes Process Engine — Self-Exciting Point Process for Cascading Failures.
============================================================================

Mathematical Framework:
    λ(t) = μ + Σ_{t_k < t} α · exp(-β · (t - t_k))

Where:
    μ  ≥ 0 : baseline (background) failure rate
    α  ≥ 0 : excitation strength (jump size when an event occurs)
    β  > 0 : decay rate (how quickly excitation fades)

Stationary iff branching ratio α/β < 1.

Failure Probability (Poisson link):
    p(t) = 1 - exp(-λ(t) · Δt)

When α = 0: reduces to constant-rate Bernoulli (backward compatible).

MLE Fitting:
    ℓ(μ, α, β) = Σ log λ(t_i) - ∫_0^T λ(s) ds
    Integral closed form: μT + (α/β) Σ [1 - exp(-β(T - t_i))]
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class HawkesParams:
    """
    Parameters for a univariate Hawkes process.

    Attributes
    ----------
    mu : float
        Baseline (background) intensity. Must be ≥ 0.
    alpha : float
        Excitation strength (jump size per event). Must be ≥ 0.
    beta : float
        Decay rate of excitation. Must be > 0.
    """
    mu: float = 0.1
    alpha: float = 0.6
    beta: float = 1.0

    def __post_init__(self):
        if self.mu < 0:
            raise ValueError(f"mu must be >= 0, got {self.mu}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")

    @property
    def branching_ratio(self) -> float:
        """α/β — must be < 1 for stationarity."""
        return self.alpha / self.beta

    @property
    def is_stationary(self) -> bool:
        return self.branching_ratio < 1.0

    def to_dict(self) -> dict:
        return {'mu': self.mu, 'alpha': self.alpha, 'beta': self.beta}

    @classmethod
    def from_dict(cls, d: dict) -> 'HawkesParams':
        return cls(mu=d['mu'], alpha=d['alpha'], beta=d['beta'])


class HawkesProcess:
    """
    Univariate Hawkes process simulator with shared event history.

    The process maintains a running list of event times. Each call to
    `failure_probability(t)` computes λ(t) from the full history,
    and `record_event(t)` appends to the history so that excitation
    carries over across chaos stages in the pipeline.

    Parameters
    ----------
    params : HawkesParams
        Process parameters (μ, α, β).
    seed : int
        Random seed for simulation reproducibility.
    dt : float
        Discrete timestep interval (default 1.0 for daily data).
    """

    def __init__(
        self,
        params: HawkesParams,
        seed: int = 42,
        dt: float = 1.0,
    ):
        self.params = params
        self.seed = seed
        self.dt = dt
        self.rng = np.random.RandomState(seed)

        # Shared event history (persists across chaos stages)
        self.event_times: List[float] = []

        # Full intensity trace (filled during simulate())
        self.intensity_trace: List[float] = []

    # ─── Core Intensity Computation ───────────────────────────────

    def compute_intensity(self, t: float) -> float:
        """
        Compute the conditional intensity λ(t | H_t).

        λ(t) = μ + Σ_{t_k < t} α · exp(-β · (t - t_k))
        """
        mu = self.params.mu
        alpha = self.params.alpha
        beta = self.params.beta

        if alpha == 0 or len(self.event_times) == 0:
            return mu

        # Vectorised computation over event history
        events = np.array(self.event_times)
        past_mask = events < t
        if not past_mask.any():
            return mu

        past_events = events[past_mask]
        excitation = alpha * np.sum(np.exp(-beta * (t - past_events)))

        return mu + excitation

    def failure_probability(self, t: float) -> float:
        """
        Compute failure probability at time t via the Poisson link.

        p(t) = 1 - exp(-λ(t) · Δt)

        Clamped to [0, 1] for numerical safety.
        """
        lam = self.compute_intensity(t)
        p = 1.0 - np.exp(-lam * self.dt)
        return float(np.clip(p, 0.0, 1.0))

    def record_event(self, t: float):
        """Record a failure event at time t (updates shared history)."""
        self.event_times.append(float(t))

    # ─── Simulation ───────────────────────────────────────────────

    def simulate(self, n_steps: int) -> np.ndarray:
        """
        Simulate a Hawkes-driven failure mask over n_steps timesteps.

        Returns
        -------
        np.ndarray of shape (n_steps,) : binary mask
            1 = data survives, 0 = data lost (failure occurred)
        """
        mask = np.ones(n_steps, dtype=np.float64)
        self.intensity_trace = []

        for t in range(n_steps):
            t_float = float(t)
            lam = self.compute_intensity(t_float)
            self.intensity_trace.append(lam)

            p = 1.0 - np.exp(-lam * self.dt)
            p = np.clip(p, 0.0, 1.0)

            if self.rng.uniform() < p:
                mask[t] = 0.0
                self.record_event(t_float)

        return mask

    def simulate_2d(self, n_rows: int, n_cols: int) -> np.ndarray:
        """
        Simulate a 2D failure mask (e.g., items × timesteps).

        Each column (timestep) shares the same Hawkes intensity,
        but individual row failures are drawn independently.

        Returns
        -------
        np.ndarray of shape (n_rows, n_cols) : binary mask
        """
        mask = np.ones((n_rows, n_cols), dtype=np.float64)
        self.intensity_trace = []

        for t in range(n_cols):
            t_float = float(t)
            lam = self.compute_intensity(t_float)
            self.intensity_trace.append(lam)

            p = 1.0 - np.exp(-lam * self.dt)
            p = np.clip(p, 0.0, 1.0)

            # Draw failures independently across rows
            failures = self.rng.uniform(size=n_rows) < p
            mask[failures, t] = 0.0

            # If ANY failure occurred at this timestep, record as event
            if failures.any():
                self.record_event(t_float)

        return mask

    # ─── Intensity Scaling ────────────────────────────────────────

    def intensity_scale_factor(self, t: float) -> float:
        """
        Ratio of current intensity to baseline: λ(t) / μ.

        Used to scale noise magnitude proportionally to cascade intensity.
        Returns 1.0 if μ = 0 (degenerate case).
        """
        if self.params.mu <= 0:
            return 1.0
        return self.compute_intensity(t) / self.params.mu

    # ─── Accessors ────────────────────────────────────────────────

    def get_intensity_trace(self) -> np.ndarray:
        """Return the full λ(t) trace from the last simulation."""
        return np.array(self.intensity_trace)

    def get_summary_stats(self) -> dict:
        """Summary statistics of the intensity trace and event history."""
        trace = self.get_intensity_trace()
        stats = {
            'n_events': len(self.event_times),
            'lambda_mean': float(np.mean(trace)) if len(trace) > 0 else self.params.mu,
            'lambda_max': float(np.max(trace)) if len(trace) > 0 else self.params.mu,
            'lambda_min': float(np.min(trace)) if len(trace) > 0 else self.params.mu,
            'lambda_final': float(trace[-1]) if len(trace) > 0 else self.params.mu,
            'branching_ratio': self.params.branching_ratio,
            'is_stationary': self.params.is_stationary,
        }
        return stats

    def reset(self, keep_params: bool = True):
        """Reset event history and intensity trace."""
        self.event_times = []
        self.intensity_trace = []
        if not keep_params:
            self.rng = np.random.RandomState(self.seed)

    def save_trace(self, path: str):
        """Save intensity trace and event list to npz file."""
        np.savez(
            path,
            intensity_trace=np.array(self.intensity_trace),
            event_times=np.array(self.event_times),
            params=np.array([self.params.mu, self.params.alpha, self.params.beta]),
        )

    @staticmethod
    def load_trace(path: str) -> dict:
        """Load saved trace data."""
        data = np.load(path)
        return {
            'intensity_trace': data['intensity_trace'],
            'event_times': data['event_times'],
            'params': data['params'],
        }


# ═══════════════════════════════════════════════════════════════
# MLE Fitting
# ═══════════════════════════════════════════════════════════════

def hawkes_log_likelihood(
    params_vec: np.ndarray,
    event_times: np.ndarray,
    T: float,
) -> float:
    """
    Negative log-likelihood for a univariate Hawkes process.

    ℓ(μ, α, β) = Σ log λ(t_i) - ∫_0^T λ(s) ds

    Returns NEGATIVE log-likelihood (for minimisation).
    """
    mu, alpha, beta = params_vec
    n = len(event_times)

    if n == 0:
        return mu * T

    times = np.sort(event_times)

    # Term 1: Σ log λ(t_i)
    log_lam_sum = 0.0
    for i in range(n):
        t_i = times[i]
        lam_i = mu
        if i > 0:
            past = times[:i]
            lam_i += alpha * np.sum(np.exp(-beta * (t_i - past)))

        if lam_i <= 1e-15:
            lam_i = 1e-15
        log_lam_sum += np.log(lam_i)

    # Term 2: ∫_0^T λ(s) ds = μT + (α/β) Σ [1 - exp(-β(T - t_i))]
    integral = mu * T
    if alpha > 0 and beta > 0:
        integral += (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))

    nll = -log_lam_sum + integral
    return nll


def fit_hawkes_mle(
    event_times: np.ndarray,
    T: float,
    initial_params: Tuple[float, float, float] = (0.1, 0.5, 1.0),
    bounds: Tuple = ((1e-6, None), (1e-6, None), (1e-3, None)),
) -> HawkesParams:
    """
    Fit Hawkes process parameters via Maximum Likelihood Estimation.
    """
    from scipy.optimize import minimize

    event_times = np.sort(np.asarray(event_times, dtype=np.float64))

    if len(event_times) < 2:
        rate = len(event_times) / T if T > 0 else 0.1
        return HawkesParams(mu=max(rate, 1e-4), alpha=0.0, beta=1.0)

    result = minimize(
        hawkes_log_likelihood,
        x0=np.array(initial_params),
        args=(event_times, T),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10},
    )

    mu_fit, alpha_fit, beta_fit = result.x

    return HawkesParams(
        mu=float(mu_fit),
        alpha=float(alpha_fit),
        beta=float(beta_fit),
    )


def fit_hawkes_from_mask(
    failure_mask: np.ndarray,
    dt: float = 1.0,
) -> HawkesParams:
    """
    Fit Hawkes parameters from a binary failure mask.
    Extracts event times from mask positions where mask == 0 (failure).
    """
    if failure_mask.ndim == 2:
        col_failures = (failure_mask == 0).any(axis=0)
        event_indices = np.where(col_failures)[0]
    else:
        event_indices = np.where(failure_mask == 0)[0]

    event_times = event_indices.astype(np.float64) * dt
    T = float(len(failure_mask) if failure_mask.ndim == 1
              else failure_mask.shape[1]) * dt

    return fit_hawkes_mle(event_times, T)
