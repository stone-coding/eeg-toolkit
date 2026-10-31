"""MockAmplifier: replays pre-recorded EEG in chunks at the real sampling rate."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Iterable

import mne
import numpy as np

from eegkit.logging import get_logger
from eegkit.realtime.buffer import RingBuffer
from eegkit.realtime.device import (
    AcquisitionDevice,
    Chunk,
    DeviceDisconnectedError,
    DeviceState,
    DeviceStats,
    StreamEndedError,
)

log = get_logger(__name__)

_END = object()  # queue sentinel: no more chunks (disconnect or end of finite stream)


class MockAmplifier(AcquisitionDevice):
    """Simulated amplifier streaming ``data`` (``(n_ch, n_samples)``, volts) in real time.

    * **Timing**: chunk *k* is released at ``t0 + k * chunk_dt`` (absolute deadlines), so
      scheduling jitter does not accumulate into drift. ``speed`` > 1 compresses time
      (handy in tests); 1.0 is real time.
    * **Ring buffer**: every delivered sample is also written to :attr:`buffer` (the device's
      recent history, e.g. for display) - independent of how fast consumers read chunks.
    * **Markers**: stimulus markers (``(sample, label)`` pairs, repeated each loop) plus
      user markers from :meth:`send_marker`; each chunk carries the markers inside it.
    * **Faults** (for testing consumers): :meth:`inject_drop`, :meth:`inject_disconnect`, and
      a bounded chunk queue that discards the *oldest* chunk when the consumer is too slow -
      like a real device FIFO, the producer never blocks.
    """

    def __init__(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: list[str],
        chunk_samples: int = 16,
        markers: Iterable[tuple[int, str]] = (),
        loop: bool = True,
        speed: float = 1.0,
        queue_chunks: int = 256,
        buffer_seconds: float = 30.0,
    ) -> None:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] != len(ch_names):
            raise ValueError("data must be (n_channels, n_samples) matching ch_names")
        if chunk_samples < 1 or speed <= 0 or sfreq <= 0:
            raise ValueError("chunk_samples, speed and sfreq must be positive")
        self._data = data
        self._sfreq = float(sfreq)
        self._ch_names = list(ch_names)
        self.chunk_samples = chunk_samples
        self.loop = loop
        self.speed = speed
        self._markers = sorted((int(s), str(lab)) for s, lab in markers)

        self.buffer = RingBuffer(len(ch_names), max(int(buffer_seconds * sfreq), chunk_samples))
        self.markers: deque[tuple[int, str]] = deque(maxlen=1000)  # delivered, for display
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_chunks)
        self._user_markers: list[str] = []
        self._drop: set[int] = set()
        self._disconnect_after: int | None = None
        self._stats = DeviceStats()
        self._state = DeviceState.IDLE
        self._lock = threading.RLock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0  # next chunk sequence number
        self._cursor = 0  # next stream sample index

    # ---- construction helpers ----------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: mne.io.BaseRaw, **kwargs: object) -> MockAmplifier:
        """Stream the EEG channels of ``raw``; left/right (or T1/T2) annotations become markers."""
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        code = {"left": 1, "right": 2}
        alias = {"T1": 1, "T2": 2, "left": 1, "right": 2}
        try:
            events, _ = mne.events_from_annotations(raw, event_id=alias, verbose="ERROR")
        except ValueError:  # no matching annotations
            events = np.empty((0, 3), int)
        label = {v: k for k, v in code.items()}
        markers = [(int(s_ - raw.first_samp), label[int(c)]) for s_, _, c in events]
        return cls(
            raw.get_data(picks=picks),
            raw.info["sfreq"],
            [raw.ch_names[i] for i in picks],
            markers=markers,
            **kwargs,  # type: ignore[arg-type]
        )

    # ---- AcquisitionDevice -------------------------------------------------------------

    @property
    def ch_names(self) -> list[str]:
        return list(self._ch_names)

    @property
    def sfreq(self) -> float:
        return self._sfreq

    @property
    def state(self) -> DeviceState:
        return self._state

    @property
    def chunk_dt(self) -> float:
        """Wall-clock seconds between chunks."""
        return self.chunk_samples / self._sfreq / self.speed

    def start(self) -> None:
        with self._lock:
            if self._state is DeviceState.STREAMING:
                return
            if self._state is DeviceState.FINISHED:
                raise StreamEndedError("finite stream already delivered")
            self._stop_evt.clear()
            self._drain_queue()
            self._disconnect_after = (
                None if self._state is DeviceState.DISCONNECTED else self._disconnect_after
            )
            self._state = DeviceState.STREAMING
            self._thread = threading.Thread(target=self._run, name="MockAmplifier", daemon=True)
            self._thread.start()
        log.info("amplifier started", extra={"sfreq": self._sfreq, "speed": self.speed})

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            if self._state is DeviceState.STREAMING:
                self._state = DeviceState.STOPPED
            self._stop_evt.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        log.info(
            "amplifier stopped",
            extra={
                "emitted": self._stats.chunks_emitted,
                "dropped": self._stats.chunks_dropped,
                "overflowed": self._stats.chunks_overflowed,
            },
        )

    def read_chunk(self, timeout: float = 0.0) -> Chunk | None:
        try:
            item = self._queue.get(timeout=timeout) if timeout > 0 else self._queue.get_nowait()
        except queue.Empty:
            if self._state is DeviceState.DISCONNECTED:
                raise DeviceDisconnectedError("device disconnected") from None
            if self._state is DeviceState.FINISHED:
                raise StreamEndedError("end of stream") from None
            return None
        if item is _END:
            if self._state is DeviceState.FINISHED:
                raise StreamEndedError("end of stream")
            raise DeviceDisconnectedError("device disconnected")
        assert isinstance(item, Chunk)
        return item

    def send_marker(self, label: str) -> None:
        with self._lock:
            self._user_markers.append(label)

    def stats(self) -> DeviceStats:
        return self._stats

    # ---- fault injection ---------------------------------------------------------------

    def inject_drop(self, chunk_indices: Iterable[int]) -> None:
        """Lose the chunks with these sequence numbers (never delivered or buffered)."""
        with self._lock:
            self._drop |= set(chunk_indices)

    def inject_disconnect(self, after_chunks: int) -> None:
        """Disconnect once ``after_chunks`` chunks have been produced in total."""
        with self._lock:
            self._disconnect_after = after_chunks

    # ---- internals ---------------------------------------------------------------------

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _push(self, item: object) -> None:
        """Enqueue without ever blocking; on a full queue the oldest chunk is discarded."""
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    old = self._queue.get_nowait()
                except queue.Empty:
                    continue
                if old is not _END:
                    self._stats.chunks_overflowed += 1

    def _next_samples(self) -> np.ndarray | None:
        n_total = self._data.shape[1]
        if not self.loop and self._cursor >= n_total:
            return None
        idx = np.arange(self._cursor, self._cursor + self.chunk_samples)
        if self.loop:
            return self._data[:, idx % n_total]
        return self._data[:, idx[idx < n_total]]

    def _markers_between(self, start: int, stop: int) -> list[tuple[int, str]]:
        n_total = self._data.shape[1]
        out: list[tuple[int, str]] = []
        loops = range(start // n_total, (stop - 1) // n_total + 1) if self.loop else [0]
        for lap in loops:
            for pos, lab in self._markers:
                g = lap * n_total + pos
                if start <= g < stop:
                    out.append((g, lab))
        return out

    def _run(self) -> None:
        t0 = time.monotonic()
        k = 0
        dt = self.chunk_dt
        while not self._stop_evt.is_set():
            wait = t0 + k * dt - time.monotonic()
            if wait > 0 and self._stop_evt.wait(wait):
                break
            with self._lock:
                if self._disconnect_after is not None and self._seq >= self._disconnect_after:
                    self._state = DeviceState.DISCONNECTED
                    self._push(_END)
                    log.warning("amplifier disconnected", extra={"seq": self._seq})
                    return
                samples = self._next_samples()
                if samples is None or samples.shape[1] == 0:
                    self._state = DeviceState.FINISHED
                    self._push(_END)
                    return
                start, n = self._cursor, samples.shape[1]
                seq = self._seq
                self._seq += 1
                self._cursor += n
                k += 1
                if seq in self._drop:  # lost on the wire: time passes, nothing arrives
                    self._stats.chunks_dropped += 1
                    continue
                marks = self._markers_between(start, start + n)
                marks += [(start, lab) for lab in self._user_markers]
                self._user_markers.clear()
                chunk = Chunk(seq, start, samples, time.monotonic(), tuple(marks))
                self._stats.chunks_emitted += 1
                self._stats.samples_emitted += n
            self.buffer.append(samples)
            self.markers.extend(marks)
            self._push(chunk)
