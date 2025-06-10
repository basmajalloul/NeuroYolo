# NeuroKinematic Analyzer (NeuroYOLO) Pipeline

This repository provides a complete workflow for building and deploying a **video-based human pose analysis platform** using YOLOv11n-pose.  
It includes scripts for keyframe extraction and annotation, dataset preparation, model fine-tuning, and a fully-featured Streamlit dashboard for real-time pose analysis in rehabilitation and movement science.

---

## 📦 Project Structure

```
videos/                   # Raw input videos (.mp4)
annotations/              # Keyframe images and COCO-format JSONs (generated)
data/                     # Final train/val split dataset + data.yaml
processed/                # Dashboard outputs and session artifacts
assets/                   # Logo and custom assets for dashboard
generate_annotations.py   # Extract keyframes, generate pose annotations
split_dataset.py          # Split dataset into train/val, merge COCO annotations
generate_data_yaml.py     # Generate data.yaml for YOLO training
fine_tune_yolov11n.ipynb  # Jupyter notebook for model fine-tuning
app.py                    # Streamlit dashboard (NeuroYOLO Analyzer)
processor.py              # Analysis and inference core for dashboard
README.md                 # You are here!
```

---

## 1. Keyframe & Annotation Generation

Extracts representative keyframes from each video and runs YOLO pose estimation to create COCO-format annotation files.

```bash
python generate_annotations.py
```

- **Input:** Place your `.mp4` files in the `videos/` folder.
- **Output:** Keyframe images and `*_annotations.json` in `annotations/`.

**Adjust `NUM_KEYFRAMES` in the script if needed.**

---

## 2. Train/Val Dataset Split

Aggregates all annotation files, splits images and annotations into training and validation sets.

```bash
python split_dataset.py
```

- **Output:**
  - `data/images/train/`, `data/images/val/`
  - `data/annotations_train.json`, `data/annotations_val.json`

---

## 3. Data Config YAML

Generates `data/data.yaml` referencing your split dataset for YOLO training.

```bash
python generate_data_yaml.py
```

---

## 4. Fine-tune YOLOv11n-pose

Train your pose model on the prepared dataset.  
**Example usage in notebook or script:**

```python
from ultralytics import YOLO
model = YOLO("yolo11n-pose.pt")  # Start from pretrained COCO model
model.train(data="data/data.yaml", epochs=100, imgsz=256)
```

- The best checkpoint will be saved as `best.pt`.

---

## 5. Analyze Videos with the Dashboard

Copy your fine-tuned `best.pt` to the dashboard directory.  
The dashboard will **automatically use this model** for all inference and analysis.

```bash
streamlit run app.py
```

- Upload coach and participant videos.
- Run fully automated pose analysis.
- Review similarity, joint deviation, smoothness metrics, annotated videos, and reports.

**To update the model:** Replace `best.pt` with a new checkpoint file.

---

## 🛠️ Requirements

- Python 3.8+
- [Ultralytics YOLOv8/YOLOv11](https://github.com/ultralytics/ultralytics)
- Streamlit
- OpenCV
- NumPy, pandas, matplotlib, seaborn, scikit-learn, dtw-python, plotly, PyYAML

Install dependencies:
```bash
pip install -r requirements.txt
```
*(See notebook/scripts for additional dependencies.)*

---

## 🚀 Quick Start

1. Place your videos in `videos/`.
2. Run `generate_annotations.py` and `split_dataset.py`.
3. Run `generate_data_yaml.py`.
4. Fine-tune your pose model (`fine_tune_yolov11n.ipynb` or custom script).
5. Copy `best.pt` into your dashboard directory.
6. Launch the dashboard:  
   ```bash
   streamlit run app.py
   ```

---

## 📄 Citation

If you use this framework, please cite:

> [BibTeX entry or manuscript reference once published.]

---

## 📝 Notes

- The number of keyframes and YOLO model paths are configurable in the scripts.
- All dataset outputs are in COCO format, compatible with Ultralytics YOLO pose training.
- The dashboard supports side-by-side playback, trend visualization, and session reports.
- See `app.py` for advanced dashboard customization.

---

**Questions, issues, or feature requests?**  
Open an issue or contact the maintainer!
