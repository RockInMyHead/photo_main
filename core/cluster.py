"""
Face clustering module for Face Sorter application.
Contains functions for face detection, feature extraction, and clustering.
"""

import random
from pathlib import Path
from typing import Dict, List, Set, Any
import json

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"}


def find_images(directory: Path) -> List[Path]:
    """Find all images in the given directory."""
    images = []
    if not directory.exists() or not directory.is_dir():
        return images
    
    for ext in IMG_EXTS:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.lower()}"))
    
    return sorted(list(set(images)))


def simulate_face_detection(image_path: Path) -> List[Dict]:
    """
    Simulate face detection for demonstration purposes.
    In a real implementation, this would use a face detection model.
    """
    # Simulate that some images have faces, some don't
    num_faces = random.choice([0, 0, 1, 1, 1, 2])  # Bias towards 1 face
    
    faces = []
    for i in range(num_faces):
        face = {
            "bbox": [100 + i*50, 100 + i*50, 200 + i*50, 200 + i*50],  # x1, y1, x2, y2
            "confidence": random.uniform(0.8, 0.99),
            "embedding": [random.uniform(-1, 1) for _ in range(512)]  # 512-dim face embedding
        }
        faces.append(face)
    
    return faces


def cluster_faces(all_faces: List[Dict], eps_sim: float = 0.55, min_samples: int = 2) -> Dict:
    """
    Cluster faces based on similarity.
    This is a simplified clustering algorithm for demonstration.
    """
    if not all_faces:
        return {"clusters": {}, "centroids": {}}
    
    # Simple clustering simulation
    clusters = {}
    centroids = {}
    
    # Group faces into random clusters for demonstration
    num_clusters = max(1, len(all_faces) // 3)  # Rough grouping
    
    for i, face in enumerate(all_faces):
        cluster_id = i % num_clusters
        
        if cluster_id not in clusters:
            clusters[cluster_id] = []
            centroids[cluster_id] = face["embedding"]
        
        clusters[cluster_id].append(face)
    
    return {"clusters": clusters, "centroids": centroids}


def build_plan(
    group_dir: Path,
    group_thr: int = 3,
    eps_sim: float = 0.55,
    min_samples: int = 2,
    min_face: int = 110,
    blur_thr: float = 45.0,
    det_size: int = 640,
    gpu_id: int = 0,
    # YOLO params
    yolo_person_gate: bool = True,
    yolo_model: str = "yolov8n.pt",
    yolo_conf: float = 0.25,
    yolo_imgsz: int = 640,
    yolo_device: str = "auto",
    yolo_half: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build processing plan for face clustering.
    
    Args:
        group_dir: Directory containing images to process
        group_thr: Minimum images per person to create a group
        eps_sim: Similarity threshold for clustering
        min_samples: Minimum samples for cluster
        min_face: Minimum face size
        blur_thr: Blur threshold
        det_size: Detection size
        gpu_id: GPU ID
        yolo_person_gate: Use YOLO for person filtering
        yolo_model: YOLO model name
        yolo_conf: YOLO confidence threshold
        yolo_imgsz: YOLO image size
        yolo_device: YOLO device
        yolo_half: Use half precision
        
    Returns:
        Dictionary with processing plan
    """
    print(f"Processing directory: {group_dir}")
    
    # Find all images
    images = find_images(group_dir)
    total_images = len(images)
    
    if total_images == 0:
        return {
            "eligible_clusters": [],
            "cluster_centroids": {},
            "cluster_images": {},
            "group_only_images": [],
            "unknown_images": [],
            "stats": {
                "images_total": 0,
                "images_unknown_only": 0,
                "images_group_only": 0
            },
        }
    
    print(f"Found {total_images} images")
    
    # Detect faces in all images
    all_faces = []
    image_to_faces = {}
    
    for img_path in images:
        faces = simulate_face_detection(img_path)
        image_to_faces[img_path] = faces
        
        for face in faces:
            face["image_path"] = str(img_path)
            all_faces.append(face)
    
    print(f"Detected {len(all_faces)} faces across {total_images} images")
    
    if not all_faces:
        # No faces detected
        return {
            "eligible_clusters": [],
            "cluster_centroids": {},
            "cluster_images": {},
            "group_only_images": [],
            "unknown_images": [str(img) for img in images],
            "stats": {
                "images_total": total_images,
                "images_unknown_only": total_images,
                "images_group_only": 0
            },
        }
    
    # Cluster faces
    clustering_result = cluster_faces(all_faces, eps_sim, min_samples)
    clusters = clustering_result["clusters"]
    centroids = clustering_result["centroids"]
    
    # Build result
    eligible_clusters = []
    cluster_centroids = {}
    cluster_images = {}
    group_only_images = []
    unknown_images = []
    
    for cluster_id, faces in clusters.items():
        if len(faces) >= group_thr:
            # Large enough cluster
            eligible_clusters.append(cluster_id)
            cluster_centroids[cluster_id] = centroids[cluster_id]
            cluster_images[cluster_id] = list(set(face["image_path"] for face in faces))
        else:
            # Small cluster - add to group_only
            for face in faces:
                img_path = face["image_path"]
                if img_path not in group_only_images:
                    group_only_images.append(img_path)
    
    # Images without faces go to unknown
    images_with_faces = set()
    for faces in image_to_faces.values():
        if faces:
            for face in faces:
                images_with_faces.add(face["image_path"])
    
    for img_path in images:
        if str(img_path) not in images_with_faces:
            unknown_images.append(str(img_path))
    
    result = {
        "eligible_clusters": eligible_clusters,
        "cluster_centroids": cluster_centroids,
        "cluster_images": cluster_images,
        "group_only_images": group_only_images,
        "unknown_images": unknown_images,
        "stats": {
            "images_total": total_images,
            "images_unknown_only": len(unknown_images),
            "images_group_only": len(group_only_images)
        },
    }
    
    print(f"Clustering result: {len(eligible_clusters)} clusters, {len(group_only_images)} group_only, {len(unknown_images)} unknown")
    
    return result


# Additional utility functions for compatibility
def get_img_extensions() -> Set[str]:
    """Get supported image extensions."""
    return IMG_EXTS
