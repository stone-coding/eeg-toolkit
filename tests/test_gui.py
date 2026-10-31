"""Headless smoke tests of the Streamlit pages (Streamlit's AppTest, no browser)."""

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def _live_app():
    import streamlit as st

    from eegkit.gui import live

    live.render()
    assert st is not None


def test_live_page_streams_synthetic_signal():
    import time

    at = st_testing.AppTest.from_function(_live_app, default_timeout=30).run()
    assert not at.exception
    at.button[0].click().run()  # Start (synthetic source is the default)
    assert not at.exception and "amp" in at.session_state
    time.sleep(1.8)
    at.run()
    assert not at.exception
    amp = at.session_state["amp"]
    assert amp.stats().chunks_emitted > 5
    labels = [m.label for m in at.metric]
    assert "Latency" in labels
    # stop cleanly
    next(b for b in at.button if b.label == "Stop").click().run()
    assert "amp" not in at.session_state and not at.exception


def _explorer_app():
    from eegkit.gui import explorer

    explorer.render()


def test_explorer_page_handles_empty_catalog(tmp_path, monkeypatch):
    import streamlit as st

    st.cache_resource.clear()  # context() is process-global; don't reuse another test's catalog
    monkeypatch.setenv("EEGKIT_CONFIG", str(tmp_path / "missing.yaml"))
    monkeypatch.chdir(tmp_path)  # defaults resolve to an empty ./outputs/catalog.sqlite
    at = st_testing.AppTest.from_function(_explorer_app, default_timeout=30).run()
    assert not at.exception
    assert any("catalog is empty" in i.value for i in at.info)
    st.cache_resource.clear()


def test_explorer_page_renders_real_catalog_artifacts(lab_env):
    """Explorer over a catalog with indexed runs and published results renders every tab."""
    import streamlit as st

    st.cache_resource.clear()
    at = st_testing.AppTest.from_function(_explorer_app, default_timeout=60).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Raw traces", "PSD", "ERD/ERS", "Topomaps"]
    assert len(at.metric) == 4
    st.cache_resource.clear()
