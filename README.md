# 🌸 Flower Image Classification with YOLO (Daisy vs. Dandelion)

An image classification project built with **Ultralytics YOLO** (`yolo26n-cls`) to classify flower images into two categories: **Daisy** and **Dandelion**. This repository contains a preprocessed dataset, model training configurations, evaluation artifacts (metric graphs and confusion matrices), trained weights, and an inference pipeline.

---

## 📌 Project Overview

This project implements deep-learning-based computer vision classification using a lightweight YOLO classification backbone (`yolo26n-cls`). The model was fine-tuned on a balanced dataset of flower images sourced from the Alexander Mamaev Flowers Recognition dataset (exported via Roboflow).

- **Task Type**: Image Classification (`classify`)
- **Classes**: 2 (`daisy`, `dandelion`)
- **Base Architecture**: `yolo26n-cls` (1.53M parameters, 3.3 GFLOPs)
- **Top-1 Validation Accuracy**: **89.01%** (achieved after 10 epochs)
- **Primary Environment**: PyTorch with CUDA acceleration (NVIDIA RTX GPU)

---

## ✨ Key Features

- **Binary Species Classification**: Reliably discriminates between daisies and dandelions.
- **Lightweight & High Speed**: Utilizes the nano classification variant of YOLO, capable of sub-20ms inference per image on modern GPUs.
- **Trained Checkpoints Included**: Best and last model weights are preserved under `runs/classify/flower_classification/weights/`.
- **Pre-structured Dataset**: Ready-to-use dataset directory split into `train` and `test` subsets with balanced classes.
- **Interactive Notebook Workflow**: Contains an end-to-end Jupyter Notebook (`Untitled.ipynb`) detailing dataset setup, model fine-tuning, and batch prediction.
- **Evaluation & Diagnostics**: Includes exported confusion matrices, normalized confusion matrices, and training curves (`results.png`, `results.csv`).

---

## 🛠️ Technologies Used

| Technology | Purpose | Verified Version |
|---|---|---|
| **Python** | Programming Language | `3.12.7` |
| **Ultralytics YOLO** | Model Training & Inference Framework | `8.4.183`+ |
| **PyTorch** (`torch` / `torchvision`) | Deep Learning Backend | `2.11.0+cu128` |
| **OpenCV** (`opencv-python`) | Image Processing | `5.0.0.93` |
| **Pillow** (`PIL`) | Image Handling & Loading | `12.3.0` |
| **Matplotlib** | Training Metric Plotting & Visualizations | `3.11.2` |
| **Roboflow / Kaggle** | Dataset Origin and Annotation Format | Folder format |

---

## 📂 Project Structure

```text
Image-classification-/
├── custom_dataset/                     # Active dataset for model training & evaluation
│   ├── train/                          # Training split (1,000 images)
│   │   ├── daisy/                      # 500 images
│   │   └── dandelion/                  # 500 images
│   └── test/                           # Test / Validation split (182 images)
│       ├── daisy/                      # 77 images
│       └── dandelion/                  # 105 images
├── valid/                              # Supplementary validation directory (364 images)
│   ├── daisy/                          # 163 images
│   └── dandelion/                      # 201 images
├── test_images/                        # Sample images for testing inference (10 images)
├── runs/                               # Experiment logs and generated outputs
│   └── classify/
│       ├── flower_classification/      # Completed 10-epoch training run
│       │   ├── weights/
│       │   │   ├── best.pt             # Best performing model weights (Top-1: 89.01%)
│       │   │   └── last.pt             # Checkpoint from final epoch
│       │   ├── confusion_matrix.png    # Classification confusion matrix
│       │   ├── confusion_matrix_normalized.png
│       │   ├── results.csv             # Epoch-by-epoch loss & accuracy metrics
│       │   ├── results.png             # Loss and accuracy curves
│       │   └── args.yaml               # Training hyperparameters snapshot
│       └── predict/                    # Visual inference outputs on test images
├── weights/
│   └── yolo26n.pt                      # Base detection weights
├── yolo26n-cls.pt                      # Pretrained YOLO nano classification base weights
├── yolo26n-cls (1).pt                  # Backup copy of pretrained base weights
├── Untitled.ipynb                      # Main Jupyter Notebook (training & batch prediction)
├── image classification.ipynb          # Secondary / template notebook
├── README.dataset.txt                  # Dataset metadata from Kaggle / Roboflow
├── README.roboflow.txt                 # Roboflow export configuration details
└── README.md                           # Repository documentation
```

---

## 📊 Dataset & Model Information

### Dataset Breakdown
- **Source**: Kaggle [Flowers Recognition Dataset](https://www.kaggle.com/alxmamaev/flowers-recognition) by Alexander Mamaev, formatted via Roboflow.
- **Dataset License**: Public Domain.
- **Input Dimensions**: Resized to `224x224` during training and inference.
- **Class Distribution**:
  - `custom_dataset/train`: 1,000 images (500 daisy, 500 dandelion) — *100% balanced*.
  - `custom_dataset/test`: 182 images (77 daisy, 105 dandelion).
  - `valid/`: 364 images (163 daisy, 201 dandelion).

### Model Architecture
- **Model Backbone**: `YOLO26n-cls`
- **Layers**: 86
- **Parameters**: 1,533,666
- **GFLOPs**: 3.3

### Training Summary (10 Epochs)
Training logs recorded in `runs/classify/flower_classification/results.csv`:

| Metric | Initial (Epoch 1) | Mid (Epoch 5) | Final (Epoch 10) |
|---|---|---|---|
| **Train Loss** | 0.6368 | 0.2829 | **0.1382** |
| **Validation Loss** | 0.4924 | 0.5157 | **0.4224** |
| **Top-1 Accuracy** | 77.47% | 87.91% | **89.01%** |
| **Top-5 Accuracy** | 100.0% | 100.0% | **100.0%** |

---

## ⚙️ Prerequisites

- **Python**: Version `3.10` to `3.12` installed.
- **Hardware**:
  - **GPU (Recommended)**: NVIDIA GPU with CUDA support (tested on RTX 3050 Laptop GPU, 6GB VRAM).
  - **CPU**: Supported, though training will take longer.
- **Git** installed on your system.

---

## 🚀 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/om-saxena34/Image-classification-.git
   cd Image-classification-
   ```

2. **Create and Activate a Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install ultralytics torch torchvision opencv-python pillow matplotlib
   ```

   *(Optional: If using CUDA, ensure PyTorch is installed with the appropriate CUDA version matching your GPU drivers from [pytorch.org](https://pytorch.org/).)*

---

## 🔧 Environment Variables and Configuration

No external cloud API keys or secret credentials are required to run this project. All training configurations are managed via function arguments or the Ultralytics CLI:

- `data`: Path to the dataset folder containing class subdirectories (`custom_dataset`).
- `imgsz`: Input image size (set to `224` for standard classification).
- `batch`: Batch size (set to `1` in the notebook to avoid GPU memory overflow on 6GB VRAM; increase if greater memory is available).
- `workers`: Number of dataloader workers (set to `0` on Windows to prevent multiprocessing issues).
- `device`: Set to `0` for CUDA GPU, or `'cpu'` for CPU execution.
- `amp`: Set to `True` for Automatic Mixed Precision to conserve VRAM.

---

## 💻 How to Run the Project

### Option A: Using the Jupyter Notebook (Recommended)
1. Launch Jupyter Notebook or VS Code:
   ```bash
   jupyter notebook
   ```
2. Open `Untitled.ipynb`.
3. Run the cells sequentially to install dependencies, train the model, or execute predictions.

### Option B: Training via Python Script
```python
import torch
from ultralytics import YOLO

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load pretrained base weights
model = YOLO("yolo26n-cls.pt")

# Train the model
results = model.train(
    data="custom_dataset",
    epochs=10,
    imgsz=224,
    batch=1,
    workers=0,
    device=0,       # Use 'cpu' if no NVIDIA GPU
    amp=True,
    name="flower_classification"
)
```

### Option C: Training via Ultralytics CLI
```bash
yolo classify train data=custom_dataset model=yolo26n-cls.pt epochs=10 imgsz=224 batch=1 workers=0 device=0 name=flower_classification
```

---

## 🔍 Usage Instructions & Inference

### Running Inference on New Images
You can test the trained model against individual images or a whole directory of images (e.g., `test_images/`):

```python
from ultralytics import YOLO

# Load the fine-tuned best weights
model = YOLO("runs/classify/flower_classification/weights/best.pt")

# Run prediction and save annotated output
results = model("test_images", save=True)

# Access predictions programmatically
for r in results:
    top1_index = r.probs.top1
    top1_label = r.names[top1_index]
    top1_conf = r.probs.top1conf.item()
    print(f"File: {r.path} -> Class: {top1_label} ({top1_conf:.2%})")
```

### Running Inference via CLI
```bash
yolo classify predict model=runs/classify/flower_classification/weights/best.pt source=test_images
```
Predicted images with confidence overlays will be saved to `runs/classify/predict/`.

---

## 📸 Example Inputs & Outputs

Below are the verified prediction results obtained from running `runs/classify/flower_classification/weights/best.pt` on the samples in `test_images/`:

| Input Image | Predicted Class | Model Confidence | Inference Speed |
|---|---|---|---|
| `1.jpg` | **daisy** | 1.00 (100%) | 13.0 ms |
| `2.jpg` | **daisy** | 1.00 (100%) | 18.9 ms |
| `3.jpg` | **daisy** | 1.00 (100%) | 18.9 ms |
| `4.jpg` | **daisy** | 0.98 (98%) | 17.6 ms |
| `5.jpg` | **dandelion** | 0.67 (67%) | 19.2 ms |
| `5547758...jpg` | **daisy** | 1.00 (100%) | 20.8 ms |
| `5673728...jpg` | **daisy** | 0.94 (94%) | 18.9 ms |
| `5794835...jpg` | **daisy** | 1.00 (100%) | 18.2 ms |
| `7355522...jpg` | **dandelion** | 0.89 (89%) | 20.8 ms |
| `8475769...jpg` | **daisy** | 0.95 (95%) | 19.4 ms |

*Average latency: ~11.2ms preprocessing, ~18.6ms inference, ~0.1ms postprocessing per 224x224 image.*

---

## 📈 Current Progress & Limitations

### Current Progress
- [x] Preprocessed folder-format dataset ready for training.
- [x] Model successfully trained for 10 epochs reaching 89.01% top-1 accuracy.
- [x] Checkpoints (`best.pt`, `last.pt`) and evaluation plots exported.
- [x] Verified batch prediction pipeline functioning on test images.

### Known Limitations
- **Two Classes Only**: The current model is trained only on Daisy and Dandelion. The full Kaggle flowers dataset also contains Roses, Sunflowers, and Tulips.
- **Batch Size Constraint**: Configured to `batch=1` during initial training runs to accommodate local GPU constraints.
- **Dataset Structure Alignment**: The `custom_dataset/` directory contains `train` and `test` folders, but does not have a dedicated `val` folder (the YOLO pipeline defaulted to `split=test` for validation).
- **Template Notebook**: `image classification.ipynb` is currently blank; the working code resides in `Untitled.ipynb`.

---

## 🔮 Future Improvements

- [ ] Add the remaining flower classes (Rose, Sunflower, Tulip) for multi-class 5-flower classification.
- [ ] Provide a clean standalone command-line script (`train.py` and `infer.py`) alongside the Jupyter notebooks.
- [ ] Add a `requirements.txt` file for one-step dependency installation.
- [ ] Build a web user interface using Gradio or Streamlit for drag-and-drop flower classification.
- [ ] Experiment with data augmentations (flips, random crops, color adjustments) to further improve validation accuracy.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

---

## 📜 License

- **Dataset**: Public Domain (provided by [Alexander Mamaev on Kaggle](https://www.kaggle.com/alxmamaev/flowers-recognition)).
- **Code / Project License**: No explicit software license file (such as MIT or Apache 2.0) is currently included in the repository.
