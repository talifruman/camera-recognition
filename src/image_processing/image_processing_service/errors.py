"""Image Processing Service error types."""

from __future__ import annotations


class ImageProcessingServiceConfigurationError(ValueError):
    """Raised when service configuration is invalid."""


class ImageProcessingServiceLifecycleError(RuntimeError):
    """Raised when service lifecycle operations are invalid."""