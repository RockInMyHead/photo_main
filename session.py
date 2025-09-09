"""
Session state management for Face Sorter application.
Contains session keys and initialization functions.
"""

import streamlit as st
from config import AppConfig


class S:
    """Session state keys."""
    parent_path = "parent_path"
    current_dir = "current_dir"
    selected_dirs = "selected_dirs"
    rename_target = "rename_target"
    queue = "queue"
    delete_target = "delete_target"
    delete_originals = "delete_originals"
    view_filter = "view_filter"
    logs = "logs"
    proc_logs = "proc_logs"

    # YOLO
    yolo_enabled = "yolo_enabled"
    yolo_person_gate = "yolo_person_gate"
    yolo_model = "yolo_model"
    yolo_conf = "yolo_conf"
    yolo_device = "yolo_device"
    yolo_imgsz = "yolo_imgsz"
    yolo_half = "yolo_half"


def init_state(cfg: AppConfig) -> None:
    """Initialize Streamlit session state with default values."""
    st.session_state.setdefault(S.parent_path, None)
    st.session_state.setdefault(S.current_dir, None)
    st.session_state.setdefault(S.selected_dirs, set())
    st.session_state.setdefault(S.rename_target, None)
    st.session_state.setdefault(S.queue, [])
    st.session_state.setdefault(S.delete_target, None)
    st.session_state.setdefault(S.view_filter, "Все")
    st.session_state.setdefault(S.logs, [])
    st.session_state.setdefault(S.proc_logs, [])

    # Config‑backed flags
    st.session_state.setdefault(S.delete_originals, bool(cfg.delete_originals))

    st.session_state.setdefault(S.yolo_enabled, bool(cfg.yolo_enabled))
    st.session_state.setdefault(S.yolo_person_gate, bool(cfg.yolo_person_gate))
    st.session_state.setdefault(S.yolo_model, cfg.yolo_model)
    st.session_state.setdefault(S.yolo_conf, float(cfg.yolo_conf))
    st.session_state.setdefault(S.yolo_device, cfg.yolo_device)
    st.session_state.setdefault(S.yolo_imgsz, int(cfg.yolo_imgsz))
    st.session_state.setdefault(S.yolo_half, bool(cfg.yolo_half))

