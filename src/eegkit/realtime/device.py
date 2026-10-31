"""Device interface: the contract any EEG amplifier driver (real or simulated) implements."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class DeviceError(Exception):
    """Base class for acquisition errors."""


class DeviceDisconnectedError(DeviceError):
    """The device went away (cable pulled, Bluetooth lost, USB reset...)."""


class StreamEndedError(DeviceError):
    """A finite stream (e.g. a replayed file with ``loop=False``) has been fully delivered."""


class DeviceState(enum.Enum):
    IDLE = "idle"
    STREAMING = "streaming"
    STOPPED = "stopped"
    DISCONNECTED = "disconnected"
    FINISHED = "finished"


@dataclass(frozen=True)
class Chunk:
    """A block of consecutive samples as delivered by the device.

    ``seq`` increases by one per chunk *produced*, so a gap between consecutive received
    chunks means chunks were lost. ``start_sample`` is the index of the first sample in the
    continuous stream; ``markers`` holds ``(stream_sample_index, label)`` events inside it.
    """

    seq: int
    start_sample: int
    data: np.ndarray  # (n_channels, n_samples), volts
    t_emit: float  # monotonic clock time the device emitted the chunk
    markers: tuple[tuple[int, str], ...] = ()

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])


@dataclass
class DeviceStats:
    chunks_emitted: int = 0
    chunks_dropped: int = 0  # lost by the device (injected faults)
    chunks_overflowed: int = 0  # discarded because the consumer fell behind
    samples_emitted: int = 0
    extra: dict[str, float] = field(default_factory=dict)


class AcquisitionDevice(ABC):
    """Abstract EEG acquisition device.

    A hardware driver (BrainProducts, g.tec, OpenBCI, LSL...) subclasses this and implements
    the same five members; consumers such as :class:`~eegkit.realtime.OnlineProcessor` and the
    GUI depend only on this interface.
    """

    @property
    @abstractmethod
    def ch_names(self) -> list[str]: ...

    @property
    @abstractmethod
    def sfreq(self) -> float: ...

    @property
    @abstractmethod
    def state(self) -> DeviceState: ...

    @abstractmethod
    def start(self) -> None:
        """Begin streaming (from IDLE, STOPPED or DISCONNECTED)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release resources. Must be idempotent."""

    @abstractmethod
    def read_chunk(self, timeout: float = 0.0) -> Chunk | None:
        """Next chunk, waiting up to ``timeout`` s (0 = don't wait); ``None`` if none arrived.

        Raises :class:`DeviceDisconnectedError` / :class:`StreamEndedError` once the already
        buffered chunks have been drained.
        """

    @abstractmethod
    def send_marker(self, label: str) -> None:
        """Insert an event marker at the current stream position."""

    @abstractmethod
    def stats(self) -> DeviceStats: ...
