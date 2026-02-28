"""Soft-ZCA whitening: isotropic preconditioning of residual stream activations."""

from spalf.whitening.covariance import OnlineCovariance
from spalf.whitening.whitener import SoftZCAWhitener

__all__ = ["SoftZCAWhitener", "OnlineCovariance"]
