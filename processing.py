"""
Processing module for Face Sorter application.
Contains main processing logic for clustering and person matching.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import streamlit as st

from cache import yolo_people_count_cached
from config import AppConfig
from index import load_index, save_index
from person_index import load_person_index, save_person_index
from utils import ensure_dir, log, safe_copy, _atomic_write

# Optional dependencies
try:
    from send2trash import send2trash
except Exception:
    send2trash = None


def _normalize_np(v):
    """Normalize numpy vector."""
    arr = np.array(v, dtype=np.float32)
    n = float(np.linalg.norm(arr) + 1e-12)
    return (arr / n).astype(np.float32)


def _update_person_proto(person: Dict, new_vec, k_max: int = 5, ema_alpha: float = 0.9):
    """Update person prototype with new vector."""
    nv = _normalize_np(new_vec).tolist()
    protos = person.get("protos", [])
    protos.append(nv)

    if len(protos) > k_max:
        X = np.stack([_normalize_np(v) for v in protos], axis=0)
        keep = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
        while len(keep) < k_max:
            dists = np.min((1.0 - X @ X[keep].T), axis=1)
            cand = int(np.argmax(dists))
            if cand not in keep:
                keep.append(cand)
        protos = [X[i].tolist() for i in keep]

    ema = person.get("ema")
    ema_np = _normalize_np(ema if ema is not None else protos[0])
    ema_np = ema_alpha * ema_np + (1.0 - ema_alpha) * _normalize_np(new_vec)
    ema_np = _normalize_np(ema_np)

    person["protos"] = protos
    person["ema"] = ema_np.tolist()
    person["count"] = int(person.get("count", 0)) + 1


def match_and_apply(group_dir: Path, plan: Dict, match_thr: float) -> Tuple[int, Set[Path]]:
    """Match clusters to existing persons and apply the results."""
    top2_margin = float(AppConfig().top2_margin)

    person_idx = load_person_index(group_dir)
    persons = person_idx.get("persons", [])

    raw_centroids = plan.get("cluster_centroids", {}) or {}
    centroids_norm: Dict[object, np.ndarray] = {}
    for cid, vec in raw_centroids.items():
        try:
            cid_int = int(cid)
        except Exception:
            cid_int = cid
        centroids_norm[cid_int] = _normalize_np(vec)

    proto_list: List[np.ndarray] = []
    proto_owner: List[int] = []
    per_thr: Dict[int, float] = {}

    for p in persons:
        try:
            num = int(p["number"])
        except Exception:
            continue
        thr_i = p.get("thr")
        if thr_i is not None:
            try:
                per_thr[num] = float(thr_i)
            except Exception:
                pass
        protos = p.get("protos") or ([] if not p.get("ema") else [p["ema"]])
        for v in protos:
            proto_list.append(_normalize_np(v))
            proto_owner.append(num)

    P = np.stack(proto_list, axis=0) if proto_list else None

    assigned: Dict[int, int] = {}
    new_nums: Dict[int, int] = {}

    existing_nums = sorted([int(d.name) for d in group_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    cur_max = existing_nums[-1] if existing_nums else 0

    eligible = [int(c) if str(c).isdigit() else c for c in plan.get("eligible_clusters", [])]

    for cid in eligible:
        c = centroids_norm.get(cid)
        if c is None:
            continue

        if P is not None and len(P) > 0:
            sims = (P @ c.astype(np.float32))

            per_person_scores: Dict[int, float] = {}
            for s, owner in zip(sims.tolist(), proto_owner):
                if owner not in per_person_scores or s > per_person_scores[owner]:
                    per_person_scores[owner] = s

            if not per_person_scores:
                best_num = None; s1 = -1.0; s2 = -1.0
            else:
                sorted_pairs = sorted(per_person_scores.items(), key=lambda x: x[1], reverse=True)
                best_num, s1 = sorted_pairs[0]
                s2 = sorted_pairs[1][1] if len(sorted_pairs) > 1 else -1.0

            thr_use = max(float(match_thr), float(per_thr.get(best_num, -1e9)))

            if (best_num is not None) and (s1 >= thr_use) and (s1 - s2 >= top2_margin):
                assigned[cid] = int(best_num)
                for p in persons:
                    try:
                        if int(p["number"]) == int(best_num):
                            _update_person_proto(p, c)
                            break
                    except Exception:
                        continue
            else:
                cur_max += 1
                new_nums[cid] = cur_max
                persons.append({"number": cur_max, "protos": [c.tolist()], "ema": c.tolist(), "count": 1, "thr": None})
        else:
            cur_max += 1
            new_nums[cid] = cur_max
            persons.append({"number": cur_max, "protos": [c.tolist()], "ema": c.tolist(), "count": 1, "thr": None})

    for cid, num in {**assigned, **new_nums}.items():
        ensure_dir(group_dir / str(num))

    copied: Set[Tuple[int, Path]] = set()
    cluster_images: Dict[object, List[str]] = {}
    for k, v in (plan.get("cluster_images", {}) or {}).items():
        try:
            cluster_images[int(k)] = v
        except Exception:
            cluster_images[k] = v

    for cid in eligible:
        num = assigned.get(cid, new_nums.get(cid))
        if num is None:
            continue
        for img in cluster_images.get(cid, []):
            pth = Path(img)
            key = (num, pth)
            if key in copied:
                continue
            try:
                safe_copy(pth, group_dir / str(num))
            except Exception:
                pass
            copied.add(key)

    for tag, dstname in (("group_only_images", "__group_only__"), ("unknown_images", "__unknown__")):
        items = plan.get(tag, []) or []
        if items:
            dst_dir = group_dir / dstname
            ensure_dir(dst_dir)
            for img in items:
                try:
                    safe_copy(Path(img), dst_dir)
                except Exception:
                    pass

    person_idx["persons"] = persons
    save_person_index(group_dir, person_idx)

    processed: Set[Path] = set()
    for cid in eligible:
        for img in cluster_images.get(cid, []):
            processed.add(Path(img))
    for img in plan.get("group_only_images", []) or []:
        processed.add(Path(img))
    for img in plan.get("unknown_images", []) or []:
        processed.add(Path(img))

    all_in_group = {f for f in group_dir.rglob("*") if f.is_file()}
    processed = processed.intersection(all_in_group)
    return len(persons), processed


def cleanup_processed_images(group_dir: Path, processed_images: Set[Path], *, delete_originals: bool = False) -> None:
    """Удаляем только из корня группы (safety) и только по флагу."""
    if not delete_originals:
        return
    for img_path in list(processed_images):
        try:
            if img_path.parent.resolve() != group_dir.resolve():
                continue
            if img_path.exists() and img_path.is_file():
                img_path.unlink()
                log(f"Удален оригинал: {img_path.name}")
        except Exception as e:
            log(f"Ошибка удаления {img_path.name}: {e}")


def process_targets(curr: Path, parent_root: Path) -> None:
    """Main processing function for targets."""
    if not st.button("▶️ Обработать", type="primary", use_container_width=True):
        return

    idx = load_index(parent_root)
    st.session_state["proc_logs"] = []

    targets = [Path(p) for p in st.session_state["queue"]]
    if not targets:
        targets = [p for p in curr.iterdir() if p.is_dir()]
    if not targets:
        st.warning("Нет целей для обработки.")
        return

    tot_total = tot_unknown = tot_group_only = 0
    tot_faces = tot_unique_people = tot_joint = 0
    tot_yolo_dets = 0
    tot_yolo_images = 0

    status = st.status("Идёт обработка…", expanded=True)
    with status:
        prog = st.progress(0, text=f"0/{len(targets)}")
        for k, gdir in enumerate(targets, start=1):
            st.write(f"Обработка: **{gdir.name}**")
            try:
                # Import here to avoid circular imports
                try:
                    from core.cluster import build_plan, IMG_EXTS
                except Exception:
                    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
                    def build_plan(*args, **kwargs):
                        return {
                            "eligible_clusters": [],
                            "cluster_centroids": {},
                            "cluster_images": {},
                            "group_only_images": [],
                            "unknown_images": [],
                            "stats": {"images_total": 0, "images_unknown_only": 0, "images_group_only": 0},
                        }

                plan = build_plan(
                    gdir,
                    group_thr=AppConfig().group_thr,
                    eps_sim=AppConfig().eps_sim,
                    min_samples=AppConfig().min_samples,
                    min_face=AppConfig().min_face,
                    blur_thr=AppConfig().blur_thr,
                    det_size=AppConfig().det_size,
                    gpu_id=AppConfig().gpu_id,
                    # YOLO params for gating inside backend
                    yolo_person_gate=bool(st.session_state.get("yolo_person_gate", True)),
                    yolo_model=str(st.session_state.get("yolo_model", "yolov8n.pt")),
                    yolo_conf=float(st.session_state.get("yolo_conf", 0.25)),
                    yolo_imgsz=int(st.session_state.get("yolo_imgsz", 640)),
                    yolo_device=st.session_state.get("yolo_device", "auto"),
                    yolo_half=bool(st.session_state.get("yolo_half", True)),
                )

                cluster_images = plan.get("cluster_images", {}) or {}
                faces_detections = sum(len(v) for v in cluster_images.values())
                unique_people_in_run = len(plan.get("eligible_clusters", []))

                # YOLO quick stats (для on-screen отчёта)
                freq: Dict[str, int] = {}
                yolo_this_group = 0
                yolo_imgs_this_group = 0
                if st.session_state.get("yolo_enabled", False):
                    try:
                        # считаем по всем картинкам из планов (cluster + group_only + unknown)
                        def _accumulate(paths: Iterable[str]) -> None:
                            nonlocal yolo_this_group, yolo_imgs_this_group
                            for pth in paths:
                                try:
                                    mtime = Path(pth).stat().st_mtime
                                except Exception:
                                    mtime = 0.0
                                cnt = yolo_people_count_cached(
                                    str(pth), mtime,
                                    st.session_state.get("yolo_model", "yolov8n.pt"),
                                    float(st.session_state.get("yolo_conf", 0.25)),
                                    st.session_state.get("yolo_device", "auto"),
                                    int(st.session_state.get("yolo_imgsz", 640)),
                                    bool(st.session_state.get("yolo_half", True)),
                                )
                                yolo_this_group += cnt
                                yolo_imgs_this_group += 1

                        for imgs in cluster_images.values():
                            for pth in imgs:
                                freq[pth] = freq.get(pth, 0) + 1
                                _accumulate([pth])
                        _accumulate((plan.get("group_only_images", []) or []) + (plan.get("unknown_images", []) or []))
                    except Exception as e:
                        st.warning(f"Ошибка YOLO обработки для {gdir.name}: {e}")
                        # Fallback без YOLO
                        for imgs in cluster_images.values():
                            for pth in imgs:
                                freq[pth] = freq.get(pth, 0) + 1
                else:
                    for imgs in cluster_images.values():
                        for pth in imgs:
                            freq[pth] = freq.get(pth, 0) + 1

                joint_images = sum(1 for v in freq.values() if v >= 2)

                persons_after, processed_images = match_and_apply(gdir, plan, match_thr=AppConfig().match_thr)
                idx["group_counts"][str(gdir)] = persons_after
                cleanup_processed_images(gdir, processed_images, delete_originals=bool(st.session_state.get("delete_originals", False)))

                tot_total += plan["stats"]["images_total"]
                tot_unknown += plan["stats"]["images_unknown_only"]
                tot_group_only += plan["stats"]["images_group_only"]
                tot_faces += faces_detections
                tot_unique_people += unique_people_in_run
                tot_joint += joint_images
                if st.session_state.get("yolo_enabled", False):
                    tot_yolo_dets += yolo_this_group
                    tot_yolo_images += yolo_imgs_this_group

                st.success(
                    f"{gdir.name}: фото={plan['stats']['images_total']}, уник.людей={unique_people_in_run}, "
                    f"детекций лиц={faces_detections}, group_only={plan['stats']['images_group_only']}, совместных={joint_images}" + (
                        f", YOLO детекций людей={yolo_this_group}, ср.на фото={yolo_this_group/max(1,yolo_imgs_this_group):.2f}"
                        if st.session_state.get("yolo_enabled", False) else ""
                    )
                )
                st.session_state["proc_logs"].append(
                    f"{gdir.name}: людей(детекции)={faces_detections}; уникальные люди={unique_people_in_run}; "
                    f"общие(group_only)={plan['stats']['images_group_only']}; совместные(>1 человек)={joint_images}"
                )
            except Exception as e:
                st.error(f"Ошибка в {gdir.name}: {e}")
                st.session_state["proc_logs"].append(f"{gdir.name}: ошибка — {e}")

            prog.progress(k / len(targets), text=f"{k}/{len(targets)}")

        status.update(label="Готово", state="complete")

    idx["global_stats"]["images_total"] += tot_total
    idx["global_stats"]["images_unknown_only"] += tot_unknown
    idx["global_stats"]["images_group_only"] += tot_group_only
    save_index(parent_root, idx)

    st.session_state["queue"] = []
    st.session_state["selected_dirs"] = set()

    # Report + downloads
    st.success("Обработка завершена.")
    st.markdown("**Сводка за прогон:**")
    st.write(f"- Людей на фото (детекции): **{tot_faces}**")
    st.write(f"- Уникальных людей (кластера): **{tot_unique_people}**")
    st.write(f"- Общих фото (group_only): **{tot_group_only}**")
    st.write(f"- Совместных фото (>1 человек): **{tot_joint}**")
    if st.session_state.get("yolo_enabled", False):
        avg = (tot_yolo_dets / max(1, tot_yolo_images)) if tot_yolo_images else 0.0
        st.write(f"- YOLO: детекций людей (всего): **{tot_yolo_dets}**")
        st.write(f"- YOLO: среднее людей на фото: **{avg:.2f}**")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tot_faces": tot_faces,
        "tot_unique_people": tot_unique_people,
        "tot_group_only": tot_group_only,
        "tot_joint": tot_joint,
        "yolo_enabled": bool(st.session_state.get("yolo_enabled", False)),
        "tot_yolo_detections": int(tot_yolo_dets),
        "tot_yolo_images": int(tot_yolo_images),
    }
    try:
        _atomic_write(parent_root / "last_run_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    except Exception:
        pass
    st.download_button("Скачать отчёт JSON", data=json.dumps(report, ensure_ascii=False, indent=2), file_name="run_report.json")

    st.markdown("**Детальные логи по группам:**")
    log_text = "\n".join(st.session_state.get("proc_logs", []))
    st.text_area("Логи", value=log_text, height=220)
    st.download_button("Скачать логи", data=log_text, file_name="proc_logs.txt")

