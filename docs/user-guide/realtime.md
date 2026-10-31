# Real-time simulation

Not a CLI command - a library layer (`eegkit.realtime`) used by the GUI and the tutorial.

```python
from eegkit.realtime import MockAmplifier, OnlineProcessor

amp = MockAmplifier.from_raw(raw, chunk_samples=16)  # 100 ms chunks at 160 Hz
proc = OnlineProcessor(amp.ch_names, amp.sfreq, window_s=1.0, hop_s=0.25)
amp.start()
proc.start(amp, on_result=lambda r: print(r.time_s, r.powers["C3"]["mu"]))
```

| Piece | Role |
|---|---|
| `AcquisitionDevice` (ABC) | `start/stop/read_chunk/send_marker/stats` - what a real driver implements |
| `MockAmplifier` | replays data at the real rate using absolute deadlines (no drift); history ring buffer; stimulus + user markers; fault injection |
| `RingBuffer` | thread-safe; read cursor with explicit overflow accounting |
| `OnlineProcessor` | common-average reference, sliding Hann periodogram, mu/beta power at C3/C4 |

## Failure handling

| Failure | Behaviour |
|---|---|
| Dropped chunk | `Chunk.seq` gap is detected; the window is **reset** (never a spectrum across a discontinuity); counted in `stats.dropped_chunks` |
| Device disconnect | buffered chunks are delivered, then `DeviceDisconnectedError`; the processor stops with `status="disconnected"`; `start()` reconnects |
| Consumer too slow | the device FIFO drops its **oldest** chunk (producer never blocks) and counts it; ring buffer counts overwritten unread samples |

Measured on real 64-channel data: ~2 ms mean latency from chunk emission to band-power result.
