"""Lifecycle helpers for the Image Processing Service."""

from __future__ import annotations

from enum import Enum


class ServiceState(Enum):
    """Service lifecycle states."""

    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    DEGRADED = "DEGRADED"