"""
Face Sorter — Мини‑проводник (Streamlit)
Main application entry point with refactored modular structure.
"""

from pathlib import Path

import streamlit as st

from config import AppConfig, load_config
from processing import process_targets
from session import S, init_state
from ui_components import (
    render_explorer,
    render_filters_and_options,
    render_footer_queue,
    render_move_panel,
    render_topbar,
)
from utils import get_network_url, pick_folder_dialog

# Page config
st.set_page_config(page_title="Face Sorter — Мини‑проводник", layout="wide")

st.markdown(
    """
<style>
  .row { display:grid; grid-template-columns: 160px 1fr 110px 170px 120px; gap:8px; align-items:center; padding:6px 8px; border-bottom:1px solid #f1f5f9;}
  .row:hover { background:#f8fafc; }
  .hdr { font-weight:600; color:#334155; border-bottom:1px solid #e2e8f0; }
  .thumbbox { width:150px; height:150px; display:flex; align-items:center; justify-content:center; border:1px solid #e5e7eb; border-radius:10px; overflow:hidden; background:#fff; }
</style>
""",
    unsafe_allow_html=True,
)

# Load configuration
CFG_BASE = Path(__file__).parent
CFG, UNKNOWN_CFG = load_config(CFG_BASE)


def main() -> None:
    """Main application function."""
    init_state(CFG)
    st.title("Face Sorter — Мини‑проводник")

    # LAN URL
    net_url = get_network_url()
    st.info(f"Сетевой URL (LAN): {net_url}")
    if st.get_option("server.baseUrlPath"):
        st.caption("Запуск за reverse proxy: baseUrlPath активен. Проверьте внешний URL.")
    try:
        st.link_button("Открыть по сети", net_url, use_container_width=True)
    except Exception:
        st.markdown(f"[Открыть по сети]({net_url})")
    st.text_input("Network URL", value=net_url, label_visibility="collapsed")

    if UNKNOWN_CFG:
        st.warning("Найдены неизвестные ключи в config.json: " + ", ".join(sorted(UNKNOWN_CFG)))

    # Folder picker
    if st.session_state[S.parent_path] is None:
        st.info("Выберите папку для работы.")
        pick_cols = st.columns([0.25, 0.75])
        with pick_cols[0]:
            if st.button("📂 ВЫБРАТЬ ПАПКУ", type="primary", use_container_width=True):
                folder = pick_folder_dialog()
                if folder:
                    st.session_state[S.parent_path] = folder
                    st.session_state[S.current_dir] = folder
                    st.rerun()
        with pick_cols[1]:
            manual = st.text_input("Или введите путь вручную", value="", placeholder="D:\Папка\Проект")
            if st.button("ОК", use_container_width=True):
                if manual and Path(manual).exists():
                    st.session_state[S.parent_path] = manual
                    st.session_state[S.current_dir] = manual
                    st.rerun()
                else:
                    st.error("Путь не существует.")
        return

    # When folder chosen
    curr = Path(st.session_state[S.current_dir]).expanduser().resolve()
    parent_root = Path(st.session_state[S.parent_path]).expanduser().resolve()

    render_topbar(curr)
    render_filters_and_options()
    render_explorer(curr)
    st.markdown("---")
    render_move_panel(curr)
    st.markdown("---")
    render_footer_queue()
    process_targets(curr, parent_root)


if __name__ == "__main__":  # pragma: no cover
    main()

