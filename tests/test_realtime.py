import threading
import time

import numpy as np
import pytest

from eegkit.realtime import (
    AcquisitionDevice,
    Chunk,
    DeviceDisconnectedError,
    DeviceState,
    DeviceStats,
    MockAmplifier,
    OnlineProcessor,
    RingBuffer,
    StreamEndedError,
)
from tests.conftest import make_synthetic_raw

SF = 160.0
NAMES = ["C3", "C4", "Cz", "Pz"]


def sine_data(seconds=20.0, a_c3=10e-6, a_c4=5e-6):
    t = np.arange(int(seconds * SF)) / SF
    d = np.zeros((4, t.size))
    d[0] = a_c3 * np.sin(2 * np.pi * 10 * t)  # mu at C3
    d[1] = a_c4 * np.sin(2 * np.pi * 20 * t)  # beta at C4
    return d


@pytest.fixture
def amps():
    made = []

    def make(data=None, **kw):
        kw.setdefault("speed", 1.0)
        amp = MockAmplifier(sine_data() if data is None else data, SF, NAMES, **kw)
        made.append(amp)
        return amp

    yield make
    for a in made:
        a.stop()


def drain(amp, n, timeout=5.0):
    out, t_end = [], time.monotonic() + timeout
    while len(out) < n and time.monotonic() < t_end:
        c = amp.read_chunk(timeout=0.2)
        if c is not None:
            out.append(c)
    return out


# ------------------------------------------------------------------ ring buffer


def test_ring_buffer_wraparound_and_latest():
    rb = RingBuffer(2, 5)
    rb.append(np.arange(8).reshape(1, 8).repeat(2, 0))
    assert len(rb) == 5 and rb.total_written == 8
    np.testing.assert_array_equal(rb.latest(3)[0], [5, 6, 7])
    np.testing.assert_array_equal(rb.latest(99)[0], [3, 4, 5, 6, 7])
    rb.append(np.array([[8], [8]]))
    np.testing.assert_array_equal(rb.latest(5)[0], [4, 5, 6, 7, 8])


def test_ring_buffer_read_new_and_overflow_accounting():
    rb = RingBuffer(1, 10)
    rb.append(np.arange(6)[None])
    np.testing.assert_array_equal(rb.read_new()[0], np.arange(6))
    assert rb.read_new().shape == (1, 0) and rb.overflow_samples == 0
    rb.append(np.arange(6, 12)[None])  # unread 6, fine
    rb.append(np.arange(12, 20)[None])  # unread 14 > capacity 10 -> 4 lost
    assert rb.overflow_samples == 4
    np.testing.assert_array_equal(rb.read_new()[0], np.arange(10, 20))


def test_ring_buffer_oversized_append_keeps_tail():
    rb = RingBuffer(1, 4)
    rb.append(np.arange(10)[None])
    np.testing.assert_array_equal(rb.latest(4)[0], [6, 7, 8, 9])
    assert rb.overflow_samples == 6


def test_ring_buffer_clear_and_validation():
    rb = RingBuffer(2, 4)
    rb.append(np.ones((2, 3)))
    rb.clear()
    assert len(rb) == 0 and rb.latest(3).shape == (2, 0) and rb.read_new().shape == (2, 0)
    rb.append(np.ones((2, 1)) * 5)
    assert len(rb) == 1
    with pytest.raises(ValueError):
        rb.append(np.ones((3, 2)))
    with pytest.raises(ValueError):
        RingBuffer(0, 5)


def test_ring_buffer_thread_safety():
    rb = RingBuffer(1, 1000)

    def writer():
        for _ in range(200):
            rb.append(np.ones((1, 10)))

    threads = [threading.Thread(target=writer) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert rb.total_written == 8000


# ------------------------------------------------------------------ mock amplifier


def test_streams_at_real_sampling_rate_without_drift(amps):
    amp = amps()
    t0 = time.monotonic()
    amp.start()
    chunks = drain(amp, 16)  # 16 chunks x 0.1 s = 1.5 s of schedule after the first
    elapsed = time.monotonic() - t0
    assert [c.seq for c in chunks] == list(range(16))
    assert 1.4 <= elapsed <= 2.0  # ideal 1.5 s
    # chunk emission times follow absolute deadlines: no accumulating lag
    emit = np.array([c.t_emit for c in chunks])
    ideal = emit[0] + np.arange(16) * amp.chunk_dt
    assert np.abs(emit - ideal).max() < 0.05
    assert amp.sfreq == SF and amp.ch_names == NAMES


def test_chunk_content_matches_source_and_buffer(amps):
    data = sine_data()
    amp = amps(data, speed=20.0)
    amp.start()
    chunks = drain(amp, 10)
    got = np.hstack([c.data for c in chunks])
    np.testing.assert_array_equal(got, data[:, : got.shape[1]])
    assert [c.start_sample for c in chunks] == [16 * i for i in range(10)]
    amp.stop()  # freeze, then the history buffer must mirror the stream exactly
    k = amp.buffer.total_written
    assert k >= 160
    np.testing.assert_array_equal(amp.buffer.latest(32), data[:, k - 32 : k])


def test_loop_wraps_and_finite_stream_ends(amps):
    short = sine_data(seconds=0.4)  # 64 samples = 4 chunks
    looped = amps(short, speed=50.0, loop=True)
    looped.start()
    c = drain(looped, 9)
    assert len(c) == 9
    np.testing.assert_array_equal(c[4].data, c[0].data)  # second lap == first lap

    finite = amps(short, speed=50.0, loop=False)
    finite.start()
    got = drain(finite, 4)
    assert len(got) == 4
    with pytest.raises(StreamEndedError):
        for _ in range(20):
            finite.read_chunk(timeout=0.1)
    assert finite.state is DeviceState.FINISHED
    with pytest.raises(StreamEndedError):
        finite.start()


def test_start_stop_idempotent_and_resumes(amps):
    amp = amps(speed=20.0)
    amp.stop()  # stop before start is harmless
    amp.start()
    amp.start()  # second start is a no-op
    assert amp.state is DeviceState.STREAMING
    first = drain(amp, 3)
    amp.stop()
    amp.stop()
    assert amp.state is DeviceState.STOPPED
    n = amp.stats().chunks_emitted
    time.sleep(0.2)
    assert amp.stats().chunks_emitted == n  # really stopped
    amp.start()
    more = drain(amp, 3)
    assert more[0].start_sample > first[-1].start_sample  # resumes, does not rewind


def test_markers_from_raw_user_and_repeat_on_loop(amps):
    raw = make_synthetic_raw(n_per_class=2)
    amp = MockAmplifier.from_raw(raw, speed=100.0, chunk_samples=32)
    amps_list = [amp]
    try:
        events = [(int(a), lab) for a, lab in amp._markers]
        assert len(events) == 4 and {lab for _, lab in events} == {"left", "right"}
        amp.send_marker("blink")
        amp.start()
        n_total = raw.n_times
        chunks = drain(amp, n_total // 32 + 40, timeout=15)
        marks = [m for c in chunks for m in c.markers]
        stim = [m for m in marks if m[1] in ("left", "right")]
        assert stim[:4] == sorted(events)
        for sample, _ in stim:  # each marker sits inside the chunk that carries it
            assert any(
                c.start_sample <= sample < c.start_sample + c.n_samples
                and (sample, _l) in c.markers
                for c in chunks
                for _l in ("left", "right")
            )
        assert any(m == (0, "blink") for m in marks)
        assert any(m[0] >= n_total for m in stim)  # repeated on the second lap
        assert len(amp.markers) > 0
    finally:
        amp.stop()
        del amps_list


def test_constructor_validation():
    with pytest.raises(ValueError):
        MockAmplifier(np.zeros((3, 10)), SF, ["a", "b"])
    with pytest.raises(ValueError):
        MockAmplifier(np.zeros((2, 10)), SF, ["a", "b"], chunk_samples=0)


# ------------------------------------------------------------------ failure handling


def test_dropped_chunks_are_visible_to_consumers(amps):
    amp = amps(speed=30.0)
    amp.inject_drop({3, 4, 9})
    amp.start()
    chunks = drain(amp, 12)
    seqs = [c.seq for c in chunks]
    assert 3 not in seqs and 4 not in seqs and 9 not in seqs
    assert amp.stats().chunks_dropped == 3
    proc = OnlineProcessor(NAMES, SF, window_s=0.5, hop_s=0.1, car=False)
    for c in chunks:
        proc.process_chunk(c)
    assert proc.stats.dropped_chunks == 3 and proc.stats.window_resets == 2


def test_device_disconnect_raises_after_buffered_chunks(amps):
    amp = amps(speed=30.0)
    amp.inject_disconnect(after_chunks=5)
    amp.start()
    got = []
    with pytest.raises(DeviceDisconnectedError):
        for _ in range(100):
            c = amp.read_chunk(timeout=0.2)
            if c:
                got.append(c)
    assert [c.seq for c in got] == [0, 1, 2, 3, 4]  # nothing buffered is lost
    assert amp.state is DeviceState.DISCONNECTED
    # reconnect: streaming resumes after the gap (the lost time shows up as a seq-contiguous
    # continuation since the device simply produced nothing while disconnected)
    amp.start()
    assert drain(amp, 2)[0].seq == 5


def test_processor_stops_cleanly_on_disconnect(amps):
    amp = amps(speed=30.0)
    amp.inject_disconnect(after_chunks=20)
    proc = OnlineProcessor(NAMES, SF, window_s=0.5, hop_s=0.1, car=False)
    amp.start()
    results = []
    proc.run(amp, results.append)  # returns by itself
    assert proc.stats.status == "disconnected" and "disconnected" in proc.stats.error
    assert len(results) > 0


def test_consumer_overflow_is_counted_and_detected(amps):
    amp = amps(speed=100.0, queue_chunks=4)
    amp.start()
    time.sleep(0.6)  # nobody reads: FIFO overflows
    amp.stop()
    st = amp.stats()
    assert st.chunks_overflowed > 10 and st.chunks_emitted > 14
    # the device never blocked: its own history buffer is full of recent data
    assert len(amp.buffer) == amp.buffer.capacity
    chunks = []
    while (c := amp.read_chunk()) is not None:
        chunks.append(c)
    assert len(chunks) == 4 and chunks[0].seq > 10  # only the newest survive
    proc = OnlineProcessor(NAMES, SF, window_s=0.2, hop_s=0.1, car=False)
    proc.process_chunk(chunks[0])
    assert proc.stats.dropped_chunks == 0  # first chunk seen: nothing to compare against
    for c in chunks[1:]:
        proc.process_chunk(c)
    assert proc.stats.dropped_chunks == 0  # survivors are contiguous


def test_overflow_between_reads_registers_gap(amps):
    amp = amps(speed=100.0, queue_chunks=4)
    amp.start()
    proc = OnlineProcessor(NAMES, SF, window_s=0.2, hop_s=0.1, car=False)
    first = drain(amp, 1)[0]
    proc.process_chunk(first)
    time.sleep(0.5)  # consumer stalls
    amp.stop()
    while (c := amp.read_chunk()) is not None:
        proc.process_chunk(c)
    assert proc.stats.dropped_chunks > 0 and proc.stats.window_resets >= 1


# ------------------------------------------------------------------ online processor


def feed(proc, data, chunk=16, seq0=0):
    out = []
    for i, s in enumerate(range(0, data.shape[1] - chunk + 1, chunk)):
        c = Chunk(seq0 + i, s, data[:, s : s + chunk], time.monotonic())
        out += proc.process_chunk(c)
    return out


def test_band_power_matches_known_sines():
    proc = OnlineProcessor(NAMES, SF, window_s=1.0, hop_s=0.25, car=False)
    res = feed(proc, sine_data(5.0))
    assert res
    p = res[-1].powers
    assert p["C3"]["mu"] == pytest.approx((10e-6) ** 2 / 2, rel=0.15)
    assert p["C3"]["beta"] < 0.02 * p["C3"]["mu"]
    assert p["C4"]["beta"] == pytest.approx((5e-6) ** 2 / 2, rel=0.15)
    assert p["C4"]["mu"] < 0.02 * p["C4"]["beta"]


def test_common_average_reference_changes_little_for_localized_signal():
    d = sine_data(5.0)
    p0 = feed(OnlineProcessor(NAMES, SF, car=False), d)[-1].powers["C3"]["mu"]
    p1 = feed(OnlineProcessor(NAMES, SF, car=True), d)[-1].powers["C3"]["mu"]
    assert p1 == pytest.approx(p0 * 0.75**2, rel=0.1)  # C3 loses 1/4 of itself (4 channels)


def test_window_and_hop_cadence():
    proc = OnlineProcessor(NAMES, SF, window_s=1.0, hop_s=0.25, car=False)
    res = feed(proc, sine_data(4.0), chunk=20)
    n = int(4.0 * SF)
    assert len(res) == (n - 160) // 40 + 1  # first full window, then one result per hop
    assert np.diff([r.end_sample for r in res]).tolist() == [40] * (len(res) - 1)
    assert res[0].time_s == pytest.approx(1.0)


def test_window_resets_after_gap_before_emitting_again():
    proc = OnlineProcessor(NAMES, SF, window_s=1.0, hop_s=0.25, car=False)
    d = sine_data(6.0)
    pre = feed(proc, d[:, : 160 * 2])  # chunks 0..19
    assert pre
    post = feed(proc, d[:, 160 * 3 : 160 * 5], seq0=25)  # chunks 20-24 lost
    assert proc.stats.dropped_chunks == 5
    # `feed` restarts sample numbering for the slice: the first post-gap result must come only
    # after a full 160-sample window of *contiguous* data has refilled
    assert post and post[0].end_sample == 160
    assert len(post) >= 3


def test_processor_validation_and_markers_passthrough():
    with pytest.raises(ValueError):
        OnlineProcessor(NAMES, SF, channels=("Fz",))
    with pytest.raises(ValueError):
        OnlineProcessor(NAMES, SF, window_s=0.5, hop_s=1.0)
    proc = OnlineProcessor(NAMES, SF, window_s=0.2, hop_s=0.1, car=False)
    d = sine_data(1.0)
    r = []
    for i in range(10):
        r += proc.process_chunk(
            Chunk(
                i,
                i * 16,
                d[:, i * 16 : (i + 1) * 16],
                time.monotonic(),
                markers=((i * 16, "left"),) if i == 5 else (),
            )
        )
    assert any(x.markers == ((80, "left"),) for x in r)


def test_near_real_time_latency_with_threads(amps):
    amp = amps()
    proc = OnlineProcessor(NAMES, SF, window_s=1.0, hop_s=0.25, car=False)
    results = []
    amp.start()
    proc.start(amp, results.append)
    time.sleep(2.2)
    proc.stop()
    amp.stop()
    assert proc.stats.status == "stopped"
    assert len(results) >= 3  # first at 1.0 s, then every 0.25 s
    assert proc.stats.max_latency_s < 0.25
    assert results[-1].powers["C3"]["mu"] > 1e-11


# ------------------------------------------------------------------ interface


class ScriptedDevice(AcquisitionDevice):
    """A different 'driver' implementing only the ABC - proves consumers are swappable."""

    def __init__(self, chunks):
        self._chunks, self._state = list(chunks), DeviceState.IDLE

    ch_names = property(lambda self: NAMES)
    sfreq = property(lambda self: SF)
    state = property(lambda self: self._state)

    def start(self):
        self._state = DeviceState.STREAMING

    def stop(self):
        self._state = DeviceState.STOPPED

    def read_chunk(self, timeout=0.0):
        if not self._chunks:
            raise StreamEndedError
        return self._chunks.pop(0)

    def send_marker(self, label):
        pass

    def stats(self):
        return DeviceStats()


def test_processor_works_with_any_device_implementation():
    d = sine_data(3.0)
    dev = ScriptedDevice(
        Chunk(i, i * 16, d[:, i * 16 : (i + 1) * 16], time.monotonic()) for i in range(30)
    )
    proc = OnlineProcessor(NAMES, SF, window_s=0.5, hop_s=0.25, car=False)
    out = []
    proc.run(dev, out.append)
    assert proc.stats.status == "finished" and out


def test_abstract_device_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AcquisitionDevice()  # type: ignore[abstract]
