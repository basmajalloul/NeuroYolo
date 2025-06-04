import os
import json
import shutil
import random

# CONFIGURATION
ANNOTATION_FOLDER = "annotations"  # Where your generated JSON files are
IMAGE_FOLDER = "annotations"  # Assuming images were saved with JSONs
OUTPUT_FOLDER = "data"  # Final dataset folder
TRAIN_RATIO = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def load_all_annotations():
    images = []
    annotations = []
    ann_id_offset = 0

    for json_file in os.listdir(ANNOTATION_FOLDER):
        if json_file.endswith(".json"):
            with open(os.path.join(ANNOTATION_FOLDER, json_file), 'r') as f:
                data = json.load(f)
                for img in data["images"]:
                    img["source_json"] = json_file  # Track where this image came from
                for ann in data["annotations"]:
                    ann["id"] += ann_id_offset  # Ensure unique annotation IDs

                images.extend(data["images"])
                annotations.extend(data["annotations"])
                ann_id_offset += len(data["annotations"])

    return images, annotations, data["categories"]

def split_and_save(images, annotations, categories):
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def filter_annotations(image_set):
        img_ids = {img["id"] for img in image_set}
        return [ann for ann in annotations if ann["image_id"] in img_ids]

    train_annotations = filter_annotations(train_images)
    val_annotations = filter_annotations(val_images)

    # Prepare folder structure
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_FOLDER, "images", split), exist_ok=True)

    # Copy images to respective folders
    for split_name, split_images in zip(["train", "val"], [train_images, val_images]):
        for img in split_images:
            src_path = os.path.join(IMAGE_FOLDER, img["file_name"])
            dst_path = os.path.join(OUTPUT_FOLDER, "images", split_name, img["file_name"])
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

    # Save annotation JSONs
    for split_name, split_images, split_annotations in zip(
        ["train", "val"], [train_images, val_images], [train_annotations, val_annotations]
    ):
        split_json = {
            "images": split_images,
            "annotations": split_annotations,
            "categories": categories
        }
        with open(os.path.join(OUTPUT_FOLDER, f"annotations_{split_name}.json"), 'w') as f:
            json.dump(split_json, f, indent=4)

    print(f"✅ Split complete: {len(train_images)} train images, {len(val_images)} validation images.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    images, annotations, categories = load_all_annotations()
    split_and_save(images, annotations, categories)