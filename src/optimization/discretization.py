"""CAGE-inspired discretization weight schedule with dynamic onset."""

import torch


class DiscretizationSchedule:
    """Discretization weight schedule with dynamic onset.

    Ramps linearly from onset_step to T_total. No disc penalty before onset.
    Onset is set dynamically when all constraints are satisfied.
    """

    def __init__(self, T_total: int, lambda_max: float = 1.0) -> None:
        self.T_total = T_total
        self.onset_step = T_total
        self.lambda_max = lambda_max

    def set_onset(self, step: int) -> None:
        """Set the step at which disc ramp begins."""
        self.onset_step = step

    def get_lambda(self, step: int) -> float:
        if step <= self.onset_step:
            return 0.0
        remaining = self.T_total - self.onset_step
        if remaining <= 0:
            return self.lambda_max
        return self.lambda_max * (step - self.onset_step) / remaining

    def state_dict(self) -> dict:
        return {"onset_step": torch.tensor(self.onset_step)}

    def load_state_dict(self, sd: dict) -> None:
        self.onset_step = int(sd["onset_step"].item())
