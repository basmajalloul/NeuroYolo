import os
import yaml

# CONFIGURATION
OUTPUT_FOLDER = "data"
DATA_YAML_PATH = os.path.join(OUTPUT_FOLDER, "data.yaml")

def generate_data_yaml():
    yaml_content = {
        "path": OUTPUT_FOLDER,
        "train": "images/train",
        "val": "images/val",
        "pose": {
            "format": "coco",
            "annotation": {
                "train": "annotations_train.json",
                "val": "annotations_val.json"
            }
        },
        "nc": 1,  # Number of classes (person)
        "keypoints": 17  # COCO-style keypoints
    }

    with open(DATA_YAML_PATH, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"✅ data.yaml generated at: {DATA_YAML_PATH}")

if __name__ == "__main__":
    generate_data_yaml()
