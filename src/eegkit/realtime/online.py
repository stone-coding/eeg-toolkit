"""Online sliding-window mu/beta band power from a device's chunk stream."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from scipy.signal import periodogram

from eegkit.analysis.spectral import band_power
from eegkit.logging import get_logger
from eegkit.realtime.buffer import RingBuffer
from eegkit.realtime.device import (
    AcquisitionDevice,
    Chunk,
    DeviceDisconnectedError,
    StreamEndedError,
)

log = get_logger(__name__)

DEFAULT_BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}


@dataclass
class PowerResult:
    """Band power for one analysis window. ``powers[channel][band]`` is in V²."""

    end_sample: int  # stream index one past the window's last sample
    time_s: float  # stream time of the window end
    powers: dict[str, dict[str, float]]
    latency_s: float  # wall time from the newest chunk's emission to this result
    markers: tuple[tuple[int, str], ...] = ()


@dataclass
class ProcessorStats:
    windows: int = 0
    dropped_chunks: int = 0
    window_resets: int = 0
    last_latency_s: float = 0.0
    max_latency_s: float = 0.0
    status: str = "idle"  # idle | running | stopped | disconnected | finished
    error: str | None = None
    latencies: list[float] = field(default_factory=list)


class OnlineProcessor:
    """Sliding-window band power at selected channels.

    Chunks go through optional common-average re-referencing (needs the full channel set),
    channel selection and a ring buffer; every ``hop_s`` of new data a Hann-windowed
    periodogram of the last ``window_s`` is integrated over each band.

    Failure handling:

    * **dropped/overflowed chunks** - detected by a gap in ``Chunk.seq``. The window is
      *reset* instead of bridged, so no spectrum is ever computed across a discontinuity; output
      resumes once a full window of contiguous data has refilled.
    * **device disconnect / end of stream** - :meth:`run` stops cleanly and records the cause in
      :attr:`stats` (``status``).
    """

    def __init__(
        self,
        ch_names: list[str],
        sfreq: float,
        channels: tuple[str, ...] = ("C3", "C4"),
        bands: dict[str, tuple[float, float]] | None = None,
        window_s: float = 1.0,
        hop_s: float = 0.25,
        car: bool = True,
    ) -> None:
        missing = [c for c in channels if c not in ch_names]
        if missing:
            raise ValueError(f"channels not in device: {missing}")
        if window_s <= 0 or hop_s <= 0 or hop_s > window_s:
            raise ValueError("require 0 < hop_s <= window_s")
        self.ch_names, self.sfreq = list(ch_names), float(sfreq)
        self.channels = tuple(channels)
        self.bands = dict(bands or DEFAULT_BANDS)
        self.window = int(round(window_s * sfreq))
        self.hop = max(1, int(round(hop_s * sfreq)))
        self.car = car
        self._pick = [ch_names.index(c) for c in channels]
        self._buf = RingBuffer(len(channels), self.window)
        self._since_hop = 0
        self._last_seq: int | None = None
        self._nfft = max(512, 1 << (self.window - 1).bit_length())
        self.stats = ProcessorStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- core (pure, synchronous) ------------------------------------------------------

    def process_chunk(self, chunk: Chunk) -> list[PowerResult]:
        """Feed one chunk; return the results (usually 0 or 1) it completed."""
        if self._last_seq is not None and chunk.seq != self._last_seq + 1:
            lost = chunk.seq - self._last_seq - 1
            self.stats.dropped_chunks += max(lost, 0)
            self.stats.window_resets += 1
            self._buf.clear()
            self._since_hop = 0
            log.warning("chunk gap, window reset", extra={"lost": lost, "seq": chunk.seq})
        self._last_seq = chunk.seq

        data = chunk.data
        if self.car:
            data = data - data.mean(axis=0, keepdims=True)
        self._buf.append(data[self._pick])
        self._since_hop += chunk.n_samples

        results: list[PowerResult] = []
        if len(self._buf) >= self.window and self._since_hop >= self.hop:
            # one result per chunk (a window is only meaningful at chunk boundaries); carry the
            # remainder so the long-run cadence stays one result per ``hop`` samples
            results.append(self._compute(chunk))
            self._since_hop %= self.hop
        return results

    def _compute(self, chunk: Chunk) -> PowerResult:
        x = self._buf.latest(self.window)
        freqs, psd = periodogram(
            x, fs=self.sfreq, window="hann", nfft=self._nfft, detrend="linear", axis=-1
        )
        powers = {
            ch: {b: float(band_power(psd[i], freqs, rng)) for b, rng in self.bands.items()}
            for i, ch in enumerate(self.channels)
        }
        end = chunk.start_sample + chunk.n_samples
        latency = time.monotonic() - chunk.t_emit
        st = self.stats
        st.windows += 1
        st.last_latency_s = latency
        st.max_latency_s = max(st.max_latency_s, latency)
        st.latencies.append(latency)
        del st.latencies[:-500]
        return PowerResult(end, end / self.sfreq, powers, latency, chunk.markers)

    # ---- threaded driver ---------------------------------------------------------------

    def run(
        self,
        device: AcquisitionDevice,
        on_result: Callable[[PowerResult], None] | None = None,
        read_timeout: float = 0.1,
    ) -> None:
        """Blocking loop: read chunks until stopped, disconnected or the stream ends."""
        self.stats.status = "running"
        try:
            while not self._stop.is_set():
                chunk = device.read_chunk(timeout=read_timeout)
                if chunk is None:
                    continue
                for res in self.process_chunk(chunk):
                    if on_result:
                        on_result(res)
            self.stats.status = "stopped"
        except DeviceDisconnectedError as exc:
            self.stats.status, self.stats.error = "disconnected", str(exc)
            log.warning("processor stopping: device disconnected")
        except StreamEndedError:
            self.stats.status = "finished"
        except Exception as exc:  # noqa: BLE001 - surface any callback/processing bug in stats
            self.stats.status, self.stats.error = "error", f"{type(exc).__name__}: {exc}"
            log.error("processor crashed", extra={"error": self.stats.error})

    def start(
        self, device: AcquisitionDevice, on_result: Callable[[PowerResult], None] | None = None
    ) -> None:
        """Run in a background thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run, args=(device, on_result), name="OnlineProcessor", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
