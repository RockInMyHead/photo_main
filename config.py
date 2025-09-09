"""
Configuration module for Face Sorter application.
Contains application configuration, settings loading and validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple


ALLOWED_CFG_KEYS = {
    "group_thr",
    "eps_sim",
    "min_samples",
    "min_face",
    "blur_thr",
    "det_size",
    "gpu_id",
    "match_thr",
    "top2_margin",
    "per_person_min_obs",
    "min_det_score",
    "min_quality",
    "delete_originals",
}


@dataclass(frozen=True)
class AppConfig:
    """Application configuration with default values."""
    group_thr: int = 3
    eps_sim: float = 0.55
    min_samples: int = 2
    min_face: int = 110
    blur_thr: float = 45.0
    det_size: int = 640
    gpu_id: int = 0
    match_thr: float = 0.44  # согласовано с клампом
    top2_margin: float = 0.08
    per_person_min_obs: int = 10
    min_det_score: float = 0.50
    min_quality: float = 0.50
    delete_originals: bool = False

    # UI‑переключатели (YOLO)
    yolo_enabled: bool = False
    yolo_person_gate: bool = True
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.25
    yolo_device: str = "auto"  # auto|cpu|cuda:0
    yolo_imgsz: int = 640
    yolo_half: bool = True


def _clamp(v, lo, hi, default):
    """Clamp value to specified range with type conversion."""
    try:
        v = type(default)(v)
        return max(lo, min(hi, v))
    except Exception:
        return default


def load_config(base: Path) -> Tuple[AppConfig, List[str]]:
    """Загрузка config.json с валидацией и клампами."""
    p = base / "config.json"
    data = asdict(AppConfig())
    unknown: List[str] = []
    if p.exists():
        try:
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                for k, v in user_cfg.items():
                    if k not in ALLOWED_CFG_KEYS and k not in AppConfig.__annotations__:
                        unknown.append(k)
                    data[k] = v
        except Exception:
            pass

    # Клампы
    data["eps_sim"] = _clamp(data["eps_sim"], 0.0, 1.0, 0.55)
    data["match_thr"] = _clamp(data["match_thr"], 0.0, 1.0, 0.44)
    data["top2_margin"] = _clamp(data["top2_margin"], 0.0, 1.0, 0.08)
    data["min_face"] = _clamp(data["min_face"], 0, 10000, 110)
    data["det_size"] = _clamp(data["det_size"], 64, 4096, 640)

    return AppConfig(**data), unknown

