"""Centralized evaluation configuration.

All evaluation/metric parameters are defined once here. Values are sourced
from the respective papers and should not be changed without understanding
the implications for comparability with published results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    """Single source of truth for all evaluation parameters.

    Frozen to prevent accidental mutation mid-benchmark.
    """

    # Global seed (all per-function seeds derived from this)
    seed: int = 42

    # --- Faithfulness: ERASER-family metrics ---
    # ERASER (DeYoung et al. 2020), NAOPC (arXiv:2408.08137), F-Fidelity (arXiv:2410.02970)
    k_values: tuple[int, ...] = (1, 5, 10, 20)

    # F-Fidelity protocol (arXiv:2410.02970 Section 3.2 & 4)
    ffidelity_beta: float = 0.5
    ffidelity_ft_epochs: int = 3
    ffidelity_ft_lr: float = 1e-5
    ffidelity_ft_batch_size: int = 16

    # Monotonicity (Arya et al. 2019)
    monotonicity_steps: int = 10

    # Normalized AOPC (arXiv:2408.08137, ACL 2025)
    naopc_beam_size: int = 15

    # Soft perturbation metrics (arXiv:2305.10496, ACL 2023)
    soft_metric_n_samples: int = 20

    # --- Adversarial sensitivity (arXiv:2409.17774) ---
    adversarial_mcp_threshold: float = 0.7
    adversarial_max_changes: int = 3
    adversarial_test_samples: int = 50

    # --- Concept analysis (arXiv:2412.07992, CB-LLM) ---
    concept_top_k_values: tuple[int, ...] = (10, 50, 100, 500)
    concept_intervention_trials: int = 50
    concept_top_n: int = 50

    # --- Baseline method accuracy parameters ---
    ig_n_steps: int = 50        # Integrated Gradients (Sundararajan et al. 2017)
    lime_num_samples: int = 500  # LIME (Ribeiro et al. 2016)

    # --- Statistical ---
    multi_seed_seeds: tuple[int, ...] = (42, 123, 456)

    # --- Display (not methodological) ---
    display_top_n_concepts: int = 20
    linear_probe_max_iter: int = 1000

    @property
    def k_max(self) -> int:
        """Maximum k, used for explanation generation and NAOPC."""
        return max(self.k_values)

    @property
    def k_display(self) -> int:
        """Middle k value for single-number table display."""
        return self.k_values[len(self.k_values) // 2]
