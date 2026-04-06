"""
__init__.py for chaos module — Hawkes-extended chaos engineering.
"""
from .perturbations import (
    DemandShock, SupplyDisruption, PriceVolatility,
    CalendarShift, GraphCorruption, AdversarialAttack,
)
from .hawkes_process import HawkesProcess, HawkesParams, fit_hawkes_mle, fit_hawkes_from_mask
from .engine import ChaosEngine
from .metrics import ResilienceMetrics
