"""Error types for the Frame Ingestion Gateway runtime."""

from __future__ import annotations


class GatewayError(Exception):
    """Base exception for gateway-specific failures."""


class GatewayLifecycleError(GatewayError):
    """Raised when gateway lifecycle rules are violated."""


class StructuralValidationError(GatewayError):
    """Raised when required ingress fields are missing or invalid."""


class UnsupportedSourceFormatError(GatewayError):
    """Raised when source_format is unsupported or non-canonical."""


class PayloadSizeMismatchError(GatewayError):
    """Raised when payload size does not match declared dimensions."""


class FrameDecodeError(GatewayError):
    """Raised when encoded payload decoding fails."""


class FrameNormalizationError(GatewayError):
    """Raised when normalized output cannot meet canonical requirements."""


class SinkEnqueueRejectedError(GatewayError):
    """Raised when sink.enqueue rejects a packet."""


class SinkUnavailableError(GatewayError):
    """Raised when sink is unavailable during publication."""


class SinkBackpressureError(GatewayError):
    """Raised when sink reports backpressure."""


class DependencyUnavailableError(GatewayError):
    """Raised when required image dependencies are not available."""
