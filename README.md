# 🧠 Brain Tumor Detection using YOLOv8 (Object Detection & Segmentation)

This project focuses on detecting **brain tumors** from MRI images using **YOLOv8n** and **YOLOv8m** segmentation/object detection models. The aim is to build a fast, lightweight model and a more accurate, larger model — then compare their performance visually and numerically.

---

## 🧩 Preprocessing & Mask Conversion

Before training, the dataset contained **MRI images + mask images**.
Masks were converted into **segmentation contours** in YOLO format.

This process allowed the model to understand the **exact tumor boundaries**, not just bounding boxes.

### ✔️ Steps Done Before Training

* Loaded original MRI images
* Loaded corresponding mask images
* Extracted the **contour area** of the tumor from each mask
* Converted every mask into YOLO segmentation format (polygon points)
* Saved the new dataset ready for YOLOv8 segmentation training

This gave much more accurate results because YOLO learned the **true shape** of the tumor.

---

## 🚀 Project Overview

Brain tumor detection and segmentation are critical for early diagnosis. Using **Ultralytics YOLOv8**, this project provides:

* Fast real‑time tumor detection
* Segmentation masks for precise localization
* Comparison between small (nano) and medium models
* Clear visualization for both models

---

## 📦 Models Trained

### **1️⃣ YOLOv8n (Nano Model)**

* **Size:** 6.5 MB
* **Mask mAP50:** 89.0%
* **Precision:** 89.5%
* **Recall:** 81.2%
* **Speed:** ⚡ 2.0 ms

📌 **Use Case:** Real‑time, low‑latency detection for edge devices.

### **2️⃣ YOLOv8m (Medium Model)**

* **Size:** 52.3 MB
* **Mask mAP50:** 92.7% ✨
* **Precision:** 92.5%
* **Recall:** 86.6%
* **Speed:** 10.1 ms

📌 **Use Case:** High‑accuracy detection for clinical/analysis systems.

---

## 🖼️ Model Comparison (Example MRI Images)

Below you can insert two image samples for comparison:

### **YOLOv8n Output**

<img width="1495" height="754" alt="image" src="https://github.com/user-attachments/assets/59041757-8cea-420c-ac7f-4ec87fe08f78" />


### **YOLOv8m Output**

<img width="1490" height="756" alt="image" src="https://github.com/user-attachments/assets/b1109a73-239b-4fc5-825d-a5c50934e714" />


### 🔍 Analysis

* YOLOv8m provides **sharper segmentation boundaries**.
* Fewer false positives than YOLOv8n.
* Slightly slower, but worth it for higher accuracy.

---

## 🧠 Training

You can train using:

```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

results = model.train(
    data='Brain_Tumor_YOLO/data.yaml',
    
    # Epochs
    epochs=100,  # increased from 50 to 100 (longer training)
    imgsz=512,
    
    # Learning rate - very important
    lr0=0.001,      # better initial learning rate
    lrf=0.0001,     # final learning rate
    
    # Batch size - important for accuracy
    batch=32,       # bigger batch = better results
    
    # Optimizer and momentum
    optimizer='SGD',      # better than default
    momentum=0.937,       # recommended value
    weight_decay=0.0005,  # light regularization
    
    # Data Augmentation - light and safe
    hsv_h=0.005,      # very slight hue change
    hsv_s=0.2,        # very slight saturation change
    hsv_v=0.15,       # very slight brightness change
    
    degrees=10,       # small rotation (10 degrees only)
    translate=0.05,   # very small translation
    scale=0.1,        # light scaling (10% only)
    flipud=0.2,       # rare vertical flip
    fliplr=0.2,       # rare horizontal flip
    
    mosaic=0.8,       # moderate mosaic
    mixup=0.01,       # very slight mixup
    
    # Early Stopping
    patience=20,      # stop after 20 epochs without improvement
    
    # Other
    device=0,
    workers=4,
    save=True,
    verbose=True,
    
    # Optional:
    # resume=True,  # resume if stopped accidentally
)
```

---

## 📈 Evaluation

After training, the `.pt` model weights are saved here:

```
/Brain_Tumor_Models/yolov8n_brain_tumor.pt
/Brain_Tumor_Models/yolov8m_brain_tumor.pt
```

You can run inference using:

```python
results = model('image.jpg')
results.show()
```

---

## 📊 Results Summary

| Model       | Size    | mAP50     | Precision | Recall    | Speed   |
| ----------- | ------- | --------- | --------- | --------- | ------- |
| **YOLOv8n** | 6.5 MB  | 89.0%     | 89.5%     | 81.2%     | ⚡ 2.0ms |
| **YOLOv8m** | 52.3 MB | **92.7%** | 92.5%     | **86.6%** | 10.1ms  |

📌 **Conclusion:** YOLOv8m performs significantly better but YOLOv8n is ideal for resource‑limited devices.

---

🖥️ Deployment (Streamlit + Gradio)

To make the model easy to use for anyone, I built an interactive web app using:

✔️ Gradio Interface

Simple UI for uploading MRI images

Displays detection + segmentation masks

Shows confidence scores

Runs locally or on Colab

✔️ Streamlit App

More advanced dashboard-style interface

Real‑time inference

Cleaner visualization of tumor boundaries

Suitable for deployment or demo

Both provide a smooth experience for doctors, students, or researchers to test the model.

```
## 📁 Project Structure

```
📦 Brain-Tumor-Detection-YOLO
│── 📂 dataset
│── 📂 runs
│── 📂 Brain_Tumor_Models
│── ├── yolov8n_brain_tumor.pt
│── ├── yolov8m_brain_tumor.pt
│── data.yaml
│── train.ipynb
│── README.md
```

---

## 🎯 Future Improvements

* Use YOLOv9 or YOLOv10 for better segmentation
* Add Grad-CAM heatmaps for interpretability
* Convert model to TensorRT for edge devices

---


## ⭐ Contribute

If you liked this project, consider giving it a ⭐ on GitHub!

---
