"""
Zephyr – Shared UI helpers
Small helpers reused across multiple dashboard tabs.
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st


def render_segmented(
    options: List[str],
    default: Optional[str] = None,
    label: str = "Section",
) -> str:
    """
    Render a segmented control with a graceful ``st.radio`` fallback for
    Streamlit versions without ``st.segmented_control``.
    """
    if not options:
        raise ValueError("options must be a non-empty list")

    chosen = default if default in options else options[0]

    try:
        choice = st.segmented_control(label, options=options, default=chosen)
    except Exception:
        choice = st.radio(label, options=options, index=options.index(chosen), horizontal=True)

    return choice or chosen