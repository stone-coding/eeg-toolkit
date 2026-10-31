"""Streamlit entry point:  ``streamlit run src/eegkit/gui/app.py``  (or ``eegkit app``)."""

import streamlit as st

from eegkit.gui import explorer, live

st.set_page_config(page_title="eegkit", page_icon="🧠", layout="wide")
page = st.navigation(
    [
        st.Page(explorer.render, title="Explorer", icon="🔎", url_path="explorer", default=True),
        st.Page(live.render, title="Live monitor", icon="📈", url_path="live"),
    ]
)
page.run()
