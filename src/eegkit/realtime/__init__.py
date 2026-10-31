"""Real-time acquisition simulation: device interface, mock amplifier, online band power."""

from eegkit.realtime.buffer import RingBuffer
from eegkit.realtime.device import (
    AcquisitionDevice,
    Chunk,
    DeviceDisconnectedError,
    DeviceError,
    DeviceState,
    DeviceStats,
    StreamEndedError,
)
from eegkit.realtime.mock import MockAmplifier
from eegkit.realtime.online import OnlineProcessor, PowerResult

__all__ = [
    "AcquisitionDevice",
    "Chunk",
    "DeviceDisconnectedError",
    "DeviceError",
    "DeviceState",
    "DeviceStats",
    "MockAmplifier",
    "OnlineProcessor",
    "PowerResult",
    "RingBuffer",
    "StreamEndedError",
]
