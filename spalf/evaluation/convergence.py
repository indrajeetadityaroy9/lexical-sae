"""Online mean estimation with relative standard error convergence."""

import math


class WelfordMean:
    """Welford's online algorithm for running mean + variance."""

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, value: float) -> None:
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (value - self._mean)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def relative_se(self) -> float:
        if self._n < 3 or abs(self._mean) < 1e-12:
            return float("inf")
        return math.sqrt(self._M2 / (self._n * (self._n - 1))) / abs(self._mean)

    @property
    def n(self) -> int:
        return self._n
