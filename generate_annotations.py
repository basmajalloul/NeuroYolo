import os
import cv2
import json
import numpy as np
from processor import select_keyframes_kmeans
from ultralytics import YOLO

# CONFIGURATION
VIDEO_FOLDER = "videos"  # Folder containing input videos
OUTPUT_FOLDER = "annotations"  # Where to save COCO JSON files
MODEL_PATH = "yolo11n-pose.pt"
NUM_KEYFRAMES = 10  # Adjust as needed

model = YOLO(MODEL_PATH)

def convert_to_python_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def process_video(video_path, output_json_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframe_indices = np.linspace(0, total_frames - 1, NUM_KEYFRAMES, dtype=int)
    
    images = []
    annotations = []
    ann_id = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    for idx in keyframe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        result = model(frame)[0]
        keypoints = result.keypoints.xy[0].cpu().numpy() if result.keypoints is not None else []

        # Prepare COCO Keypoints
        keypoints_flat = []
        for k in range(17):  # Assuming 17 COCO keypoints
            if len(keypoints) > k:
                x, y = keypoints[k]
                v = 1  # Visible (auto-detected)
            else:
                x, y, v = 0, 0, 0  # Not labeled
            keypoints_flat.extend([float(x), float(y), int(v)])

        img_filename = f"{video_name}_frame_{idx}.jpg"
        img_height, img_width = frame.shape[:2]

        images.append({
            "id": idx,
            "file_name": img_filename,
            "height": img_height,
            "width": img_width
        })

        annotations.append({
            "id": ann_id,
            "image_id": idx,
            "category_id": 1,  # Person
            "keypoints": keypoints_flat,
            "num_keypoints": sum(1 for v in keypoints_flat[2::3] if v > 0),
            "bbox": [0, 0, img_width, img_height],  # You can replace with actual bounding box if needed
            "iscrowd": 0
        })

        ann_id += 1

        # Optionally save keyframe images for manual correction
        img_output_path = os.path.join(OUTPUT_FOLDER, img_filename)
        cv2.imwrite(img_output_path, frame)

    cap.release()

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "person",
            "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                          "left_wrist", "right_wrist", "left_hip", "right_hip",
                          "left_knee", "right_knee", "left_ankle", "right_ankle"],
            "skeleton": []
        }]
    }

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4, default=convert_to_python_types)

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for video_file in os.listdir(VIDEO_FOLDER):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(VIDEO_FOLDER, video_file)
            output_json = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(video_file)[0]}_annotations.json")
            process_video(video_path, output_json)
            print(f"✅ Annotations generated for: {video_file}")
