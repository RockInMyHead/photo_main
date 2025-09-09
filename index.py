"""
Index management for Face Sorter application.
Contains functions for loading and saving global index.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from utils import ensure_dir, _atomic_write


def load_index(parent: Path) -> Dict:
    """Load global index from JSON file."""
    ensure_dir(parent)
    p = parent / "global_index.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "group_counts": {},
        "global_stats": {"images_total": 0, "images_unknown_only": 0, "images_group_only": 0},
        "last_run": None,
    }


def save_index(parent: Path, idx: Dict) -> None:
    """Save global index to JSON file."""
    idx["last_run"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write(parent / "global_index.json", json.dumps(idx, ensure_ascii=False, indent=2))

