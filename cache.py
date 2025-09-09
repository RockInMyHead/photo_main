"""
Caching module for Face Sorter application.
Contains cached functions for image thumbnails and YOLO model operations.
"""

from io import BytesIO
from pathlib import Path

import streamlit as st

# Optional dependencies
try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


@st.cache_data(show_spinner=False, max_entries=5000, ttl=3600)
def get_thumb_bytes(path_str: str, size: int, mtime: float):
    """Get cached thumbnail bytes for an image."""
    if Image is None:
        return None
    p = Path(path_str)
    try:
        im = Image.open(p)
        if ImageOps is not None:
            im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        w, h = im.size
        if w != h:
            m = min(w, h)
            left = (w - m) // 2
            top = (h - m) // 2
            im = im.crop((left, top, left + m, top + m))
        im = im.resize((size, size))
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_yolo_model(model_name: str):
    """Get cached YOLO model with proper error handling."""
    if YOLO is None:
        st.warning("YOLO не установлен. Установите: pip install ultralytics")
        return None
    
    try:
        # Проверяем, нужно ли скачивать модель
        from pathlib import Path
        model_path = Path(model_name)
        if not model_path.exists():
            st.info(f"Загружаем YOLO модель {model_name} (это может занять время при первом запуске)...")
        
        # Загружаем модель
        model = YOLO(model_name)
        
        if not model_path.exists():
            st.success(f"Модель {model_name} загружена и кэширована")
        
        return model
    except Exception as e:
        error_msg = f"Ошибка загрузки YOLO модели {model_name}: {str(e)}"
        st.error(error_msg)
        print(error_msg)  # Логируем в консоль
        return None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20000)
def yolo_people_count_cached(
    path_str: str,
    mtime: float,
    model_name: str,
    conf: float,
    device: str,
    imgsz: int,
    half: bool,
) -> int:
    """Get cached count of people detections in image using YOLO."""
    mdl = get_yolo_model(model_name)
    if mdl is None:
        return 0
    try:
        res = mdl.predict(
            source=path_str,
            imgsz=int(imgsz),
            conf=float(conf),
            device=None if device == "auto" else device,
            half=bool(half),
            verbose=False,
        )
        if not res:
            return 0
        r0 = res[0]
        if not hasattr(r0, "boxes") or r0.boxes is None:
            return 0
        cls = r0.boxes.cls.tolist()
        people_count = sum(1 for c in cls if int(c) == 0)
        return people_count
    except Exception as e:
        # Логируем ошибку в отладочных целях
        print(f"YOLO error for {path_str}: {e}")
        return 0

