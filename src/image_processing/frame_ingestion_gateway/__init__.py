"""Frame Ingestion Gateway public package exports."""

from __future__ import annotations

from image_processing.shared.contracts import (
    EnqueueRejectReason,
    EnqueueResult,
    FramePacket,
    FramePacketSink,
)

from .config import FrameIngestionGatewayConfig
from .contracts import IngressFrameMessage, NormalizedFrameBuffer
from .errors import (
    DependencyUnavailableError,
    FrameDecodeError,
    FrameNormalizationError,
    GatewayLifecycleError,
    PayloadSizeMismatchError,
    SinkBackpressureError,
    SinkEnqueueRejectedError,
    SinkUnavailableError,
    StructuralValidationError,
    UnsupportedSourceFormatError,
)
from .gateway import FrameIngestionGateway
from .packet_builder import FramePacketBuilder
from .transport import FrameIngressTransport, InMemoryFrameIngressTransport
from .validators import FrameIngestionInputValidator, FrameValidator

StubFrameIngressTransport = InMemoryFrameIngressTransport

__all__ = [
    "DependencyUnavailableError",
    "EnqueueRejectReason",
    "EnqueueResult",
    "FrameDecodeError",
    "FrameIngestionGateway",
    "FrameIngestionGatewayConfig",
    "FrameIngestionInputValidator",
    "FrameIngressTransport",
    "FramePacket",
    "FramePacketSink",
    "FrameNormalizationError",
    "FramePacketBuilder",
    "StubFrameIngressTransport",
    "FrameValidator",
    "GatewayLifecycleError",
    "InMemoryFrameIngressTransport",
    "IngressFrameMessage",
    "NormalizedFrameBuffer",
    "PayloadSizeMismatchError",
    "SinkBackpressureError",
    "SinkEnqueueRejectedError",
    "SinkUnavailableError",
    "StructuralValidationError",
    "UnsupportedSourceFormatError",
]
