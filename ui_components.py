"""
UI components for Face Sorter application.
Contains all Streamlit UI rendering functions.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st

from cache import get_thumb_bytes
from utils import human_size, safe_move, log, list_dir

# Optional dependencies
try:
    from streamlit_sortables import sort_items
except Exception:
    sort_items = None

try:
    from send2trash import send2trash
except Exception:
    send2trash = None

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

# Define IMG_EXTS here or import from core.cluster
try:
    from core.cluster import IMG_EXTS
except Exception:
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def render_topbar(curr: Path) -> None:
    """Render top navigation bar."""
    from session import S
    parent_root = Path(st.session_state[S.parent_path]).expanduser().resolve()

    cols = st.columns([0.08, 0.12, 0.80])
    with cols[0]:
        up = None if curr == Path(curr.anchor) else curr.parent
        st.button(
            "⬆️ Вверх",
            key="up",
            disabled=(up is None),
            on_click=(lambda: st.session_state.update({S.current_dir: str(up)}) if up else None),
            use_container_width=True,
        )
    with cols[1]:
        if st.button("🔄 Обновить", use_container_width=True):
            st.rerun()
    with cols[2]:
        crumbs = list(curr.parts)
        MAX_CRUMBS = 8
        def _accum(parts: List[str]) -> List[str]:
            acc = Path(parts[0]); out = [str(acc)]
            for prt in parts[1:]:
                acc = acc / prt; out.append(str(acc))
            return out
        accum_all = _accum(crumbs)
        if len(crumbs) > MAX_CRUMBS:
            shown_parts = crumbs[:2] + ("…",) + crumbs[-(MAX_CRUMBS - 3) :]
            shown_paths = accum_all[:2] + [None] + accum_all[-(MAX_CRUMBS - 3) :]
        else:
            shown_parts, shown_paths = crumbs, accum_all
        bc_cols = st.columns(len(shown_parts))
        for i, (part, pth) in enumerate(zip(shown_parts, shown_paths)):
            with bc_cols[i]:
                if pth is None:
                    st.button("…", disabled=True, use_container_width=True, key=f"bc_dots::{i}")
                else:
                    st.button(part or "/", key=f"bc::{i}", use_container_width=True, on_click=lambda p=pth: st.session_state.update({S.current_dir: p}))

    st.markdown("---")


def render_filters_and_options() -> None:
    """Render filters and options panel."""
    from session import S
    fcols = st.columns([0.33, 0.33, 0.34])
    with fcols[0]:
        st.session_state[S.view_filter] = st.selectbox("Показывать", ["Все", "Только папки", "Только изображения"], index=["Все", "Только папки", "Только изображения"].index(st.session_state[S.view_filter]))
    with fcols[1]:
        st.session_state[S.delete_originals] = st.checkbox("Удалять оригиналы после копирования (корень группы)", value=st.session_state[S.delete_originals])
    with fcols[2]:
        try:
            from ultralytics import YOLO
            yolo_available = True
        except Exception:
            yolo_available = False
        
        st.session_state[S.yolo_enabled] = st.checkbox("YOLO: детектировать людей (Ultralytics)", value=st.session_state[S.yolo_enabled], disabled=(not yolo_available))
        if not yolo_available:
            st.caption("Ultralytics не установлен. pip install ultralytics")
        else:
            ycols = st.columns(2)
            with ycols[0]:
                st.session_state[S.yolo_model] = st.selectbox("Модель", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"], index=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"].index(st.session_state[S.yolo_model]) if st.session_state[S.yolo_model] in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"] else 0)
            with ycols[1]:
                st.session_state[S.yolo_conf] = st.slider("Conf", 0.1, 0.9, float(st.session_state[S.yolo_conf]), 0.05)
            ycols2 = st.columns(3)
            with ycols2[0]:
                st.session_state[S.yolo_device] = st.selectbox("Device", ["auto", "cpu", "cuda:0"], index=["auto", "cpu", "cuda:0"].index(st.session_state[S.yolo_device]) if st.session_state[S.yolo_device] in ["auto", "cpu", "cuda:0"] else 0)
            with ycols2[1]:
                st.session_state[S.yolo_imgsz] = st.slider("imgsz", 320, 1280, int(st.session_state[S.yolo_imgsz]), 32)
            with ycols2[2]:
                st.session_state[S.yolo_half] = st.checkbox("half (FP16)", value=bool(st.session_state[S.yolo_half]))
            st.session_state[S.yolo_person_gate] = st.checkbox("YOLO-гейтинг лиц (фильтровать лица по людям)", value=st.session_state[S.yolo_person_gate])


def render_explorer(curr: Path) -> None:
    """Render file explorer."""
    from session import S
    # Header
    st.markdown('<div class="row hdr"><div>Превью</div><div>Имя</div><div>Тип</div><div>Изменён</div><div>Размер</div></div>', unsafe_allow_html=True)

    with st.container(height=700):
        items = list_dir(curr)
        vf = st.session_state[S.view_filter]
        if vf == "Только папки":
            items = [i for i in items if i.is_dir()]
        elif vf == "Только изображения":
            items = [i for i in items if i.is_file() and i.suffix.lower() in IMG_EXTS]

        for item in items:
            is_dir = item.is_dir()
            c1, c2, c3, c4, c5 = st.columns([0.14, 0.58, 0.12, 0.14, 0.10])

            # Preview
            with c1:
                if not is_dir and item.suffix.lower() in IMG_EXTS:
                    try:
                        data = get_thumb_bytes(str(item), 150, item.stat().st_mtime)
                    except Exception:
                        data = None
                    if data:
                        st.image(data)
                    else:
                        st.image(str(item), width=150)
                else:
                    st.markdown('<div class="thumbbox">📁</div>' if is_dir else '<div class="thumbbox">🗎</div>', unsafe_allow_html=True)

            # Name + inline icons
            with c2:
                icon = "📁" if is_dir else "🗎"
                name_cols = st.columns([0.72, 0.10, 0.10, 0.08])
                with name_cols[0]:
                    if is_dir:
                        if st.button(f"{icon} {item.name}", key=f"open::{item}", use_container_width=True):
                            st.session_state[S.current_dir] = str(item)
                            st.rerun()
                    else:
                        st.write(f"{icon} {item.name}")
                with name_cols[1]:
                    if is_dir:
                        checked = st.checkbox("Выбрать", key=f"sel::{item}", value=(str(item) in st.session_state[S.selected_dirs]), help="В очередь", label_visibility="collapsed")
                        if checked:
                            st.session_state[S.selected_dirs].add(str(item))
                        else:
                            st.session_state[S.selected_dirs].discard(str(item))
                with name_cols[2]:
                    if st.button("✏️", key=f"ren::{item}", help="Переименовать", use_container_width=True):
                        st.session_state[S.rename_target] = str(item)
                with name_cols[3]:
                    if st.button("🗑️", key=f"del::{item}", help="Удалить", use_container_width=True):
                        st.session_state[S.delete_target] = str(item)

            with c3:
                st.write("Папка" if is_dir else (item.suffix[1:].upper() if item.suffix else "Файл"))
            with c4:
                try:
                    st.write(datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
                except Exception:
                    st.write("—")
            with c5:
                if is_dir:
                    st.write("—")
                else:
                    try:
                        st.write(human_size(item.stat().st_size))
                    except Exception:
                        st.write("—")

            # Rename inline
            if st.session_state.get(S.rename_target) == str(item):
                rc1, rc2, rc3 = st.columns([0.70, 0.15, 0.15])
                with rc1:
                    new_name = st.text_input("Новое имя", value=item.name, key=f"ren_input::{item}", label_visibility="collapsed")
                with rc2:
                    if st.button("Сохранить", key=f"save::{item}", use_container_width=True):
                        try:
                            candidate = new_name.strip()
                            if not candidate:
                                st.error("Имя не может быть пустым.")
                            else:
                                invalid = set('<>:"/\|?*')
                                if any(ch in invalid for ch in candidate):
                                    st.error("Имя содержит недопустимые символы.")
                                else:
                                    new_path = item.parent / candidate
                                    if new_path.exists():
                                        st.error("Файл/папка с таким именем уже существует.")
                                    else:
                                        item.rename(new_path)
                                        st.session_state[S.rename_target] = None
                                        st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка: {e}")
                with rc3:
                    if st.button("Отмена", key=f"cancel::{item}", use_container_width=True):
                        st.session_state[S.rename_target] = None
                        st.rerun()

            # Delete confirm
            if st.session_state.get(S.delete_target) == str(item):
                dc1, dc2, dc3 = st.columns([0.70, 0.15, 0.15])
                with dc1:
                    st.markdown(f"❗ Подтвердите удаление: **{item.name}**")
                with dc2:
                    if st.button("Удалить", type="primary", key=f"confirm_del::{item}", use_container_width=True):
                        try:
                            if send2trash is not None:
                                send2trash(str(item))
                            else:
                                shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink(missing_ok=True)
                            st.session_state[S.delete_target] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка удаления: {e}")
                            log(f"Ошибка удаления {item}: {e}")
                with dc3:
                    if st.button("Отмена", key=f"cancel_del::{item}", use_container_width=True):
                        st.session_state[S.delete_target] = None


def render_move_panel(curr: Path) -> None:
    """Render file move panel."""
    from session import S
    with st.expander("Переместить файлы (Drag & Drop)", expanded=False):
        if sort_items is None:
            st.info("Для DnD установите пакет: pip install streamlit-sortables")
            return
        files_in_curr = [str(p) for p in curr.iterdir() if p.is_file()]
        subfolders = [p for p in curr.iterdir() if p.is_dir()]
        containers = [{"header": "Файлы (текущая папка)", "items": files_in_curr}] + [{"header": f.name, "items": []} for f in subfolders]
        result = sort_items(containers, multi_containers=True)
        if st.button("Применить переносы", use_container_width=True):
            if not result:
                st.warning("Нет изменений для переноса.")
                return
            header_to_dir = {f.name: f for f in subfolders}
            moves: List[tuple] = []
            for i, cont in enumerate(result):
                if i == 0:
                    continue
                target_dir = header_to_dir.get(cont.get("header", ""))
                if not target_dir:
                    continue
                for src_str in cont.get("items", []):
                    src_path = Path(src_str)
                    if src_path.exists() and src_path.is_file():
                        moves.append((src_path, target_dir))
            ok = errors = 0
            for src, dst in moves:
                try:
                    safe_move(src, dst); ok += 1
                except Exception as e:
                    st.error(f"Не удалось переместить {src.name}: {e}"); errors += 1
            st.success(f"Перемещено: {ok}, ошибок: {errors}")
            st.rerun()


def render_footer_queue() -> None:
    """Render footer with queue controls."""
    from session import S
    colA, colB, colC = st.columns([0.35, 0.35, 0.30])
    with colA:
        if st.button("➕ Добавить в очередь", type="secondary", use_container_width=True):
            added = 0
            for d in list(st.session_state[S.selected_dirs]):
                if d not in st.session_state[S.queue]:
                    st.session_state[S.queue].append(d); added += 1
            st.success(f"Добавлено в очередь: {added}")
    with colB:
        if st.button("🧹 Очистить очередь", use_container_width=True):
            st.session_state[S.queue] = []
            st.session_state[S.selected_dirs] = set()
            st.info("Очередь очищена.")
    with colC:
        st.write(f"В очереди: {len(st.session_state[S.queue])}")

