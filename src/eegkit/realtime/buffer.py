"""Thread-safe multichannel ring buffer with overflow accounting."""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Fixed-capacity ``(n_channels, capacity)`` circular buffer.

    Two ways to read:

    * :meth:`latest` - the most recent ``n`` samples, non-consuming (for display/windowing);
    * :meth:`read_new` - samples not yet consumed, advancing a read cursor. If the writer laps
      the reader, the oldest unread samples are overwritten and counted in
      :attr:`overflow_samples` (the data is lost, never silently).
    """

    def __init__(self, n_channels: int, capacity: int) -> None:
        if n_channels < 1 or capacity < 1:
            raise ValueError("n_channels and capacity must be >= 1")
        self.n_channels, self.capacity = n_channels, capacity
        self._buf = np.zeros((n_channels, capacity))
        self._total = 0  # samples ever written
        self._read = 0  # samples consumed via read_new()
        self._origin = 0  # samples written before the last clear() are not valid
        self.overflow_samples = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Number of valid samples currently held."""
        return min(self._total - self._origin, self.capacity)

    @property
    def total_written(self) -> int:
        return self._total

    def clear(self) -> None:
        with self._lock:
            self._read = self._origin = self._total  # discard all, keep counters monotonic

    def append(self, data: np.ndarray) -> None:
        """Append ``(n_channels, n)`` samples; older data is overwritten when full."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] != self.n_channels:
            raise ValueError(f"expected shape ({self.n_channels}, n), got {data.shape}")
        n = data.shape[1]
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:  # only the tail survives
                tail = data[:, -self.capacity :]
                self._buf[:] = self._place(tail, self._total + n - self.capacity)
            else:
                idx = (self._total + np.arange(n)) % self.capacity
                self._buf[:, idx] = data
            self._total += n
            unread = self._total - self._read
            if unread > self.capacity:
                self.overflow_samples += unread - self.capacity
                self._read = self._total - self.capacity

    def _place(self, tail: np.ndarray, start: int) -> np.ndarray:
        out = np.empty_like(self._buf)
        idx = (start + np.arange(self.capacity)) % self.capacity
        out[:, idx] = tail
        return out

    def _get(self, start: int, stop: int) -> np.ndarray:
        idx = np.arange(start, stop) % self.capacity
        return self._buf[:, idx].copy()

    def latest(self, n: int) -> np.ndarray:
        """The most recent ``n`` samples (fewer if not yet available), oldest first."""
        with self._lock:
            n = min(n, len(self))
            return self._get(self._total - n, self._total)

    def read_new(self) -> np.ndarray:
        """All unread samples (oldest first); advances the read cursor."""
        with self._lock:
            out = self._get(self._read, self._total)
            self._read = self._total
            return out
