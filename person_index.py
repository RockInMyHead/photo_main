"""
Person index management for Face Sorter application.
Contains functions for loading and saving person index with migration logic.
"""

import json
from pathlib import Path
from typing import Dict

from utils import _atomic_write


def load_person_index(group_dir: Path) -> Dict:
    """Load person index from JSON file with automatic migration."""
    p = group_dir / "person_index.json"
    data = {"persons": []}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return data

    persons = data.get("persons", [])
    changed = False
    for person in persons:
        if "protos" not in person:
            person["protos"] = [person.pop("proto")] if "proto" in person else []
            changed = True
        if "ema" not in person:
            person["ema"] = person["protos"][0] if person["protos"] else None
            changed = True
        if "count" not in person:
            person["count"] = max(1, len(person.get("protos", [])))
            changed = True
        if "thr" not in person:
            person["thr"] = None
            changed = True
        if "number" not in person:
            person["number"] = -1
            changed = True

    if changed:
        try:
            _atomic_write(p, json.dumps({"persons": persons}, ensure_ascii=False, indent=2))
        except Exception:
            pass

    return {"persons": persons}


def save_person_index(group_dir: Path, data: Dict) -> None:
    """Save person index to JSON file."""
    _atomic_write(group_dir / "person_index.json", json.dumps(data, ensure_ascii=False, indent=2))

