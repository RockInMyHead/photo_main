"""
Utility functions for Face Sorter application.
Contains file operations, logging, networking utilities.
"""

import socket
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import streamlit as st


def list_dir(p: Path) -> List[Path]:
    """List directory contents sorted by type and name."""
    try:
        items = list(p.iterdir())
    except Exception:
        return []
    items.sort(key=lambda x: (x.is_file(), x.name.lower()))
    return items


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, text: str) -> None:
    """Write text to file atomically."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def get_lan_ip() -> str:
    """Get local IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def get_network_url() -> str:
    """Get network URL for the Streamlit app."""
    ip = get_lan_ip()
    port = st.get_option("server.port") or 8501
    base = st.get_option("server.baseUrlPath") or ""
    base = ("/" + base.strip("/")) if base else ""
    return f"http://{ip}:{port}{base}"


def log(msg: str) -> None:
    """Log message to session state."""
    st.session_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def human_size(n: int) -> str:
    """Convert bytes to human readable format."""
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ", "ПБ"]
    size = float(n)
    for i, u in enumerate(units):
        if size < 1024 or i == len(units) - 1:
            return f"{size:.0f} {u}" if u == "Б" else f"{size:.1f} {u}"
        size /= 1024.0


def safe_copy(src: Path, dst_dir: Path) -> Path:
    """Safely copy file with name collision handling."""
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        stem, ext = src.stem, src.suffix
        i = 1
        while (dst_dir / f"{stem} ({i}){ext}").exists():
            i += 1
        dst = dst_dir / f"{stem} ({i}){ext}"
    shutil.copy2(src, dst)
    return dst


def safe_move(src: Path, dst_dir: Path) -> Path:
    """Safely move file with name collision handling."""
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        stem, ext = src.stem, src.suffix
        i = 1
        while (dst_dir / f"{stem} ({i}){ext}").exists():
            i += 1
        dst = dst_dir / f"{stem} ({i}){ext}"
    shutil.move(str(src), str(dst))
    return dst


# Native folder picker (Windows)
def pick_folder_dialog() -> Optional[str]:
    """Open native folder picker dialog. Placed here to keep main tidy."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title="Выберите папку")
        root.destroy()
        return folder
    except Exception:
        return None

