# 🍺 Draft Beer Image Classifier

A computer vision pipeline for classifying 5 Belgian draft beer brands 
from images, built with TensorFlow and deployed to Google Cloud Vertex AI.

Completed as part of the 
[ML6 coding challenge](https://bitbucket.org/ml6team/challenge-classify-draft-beer/src/master/).

---

## 🎯 Task

Classify images of 5 Belgian draft beers from glass photographs:

| Label | Beer |
|-------|------|
| 0 | Chimay Blue |
| 1 | Orval |
| 2 | Rochefort 10 |
| 3 | Westmalle Tripel |
| 4 | Westvleteren 12 |

---

## 📁 What's Mine vs What's ML6's

The data loading pipeline and Vertex AI deployment template were 
provided by ML6 as part of their coding challenge infrastructure.

**My own implementation:**
- Model architecture and fine-tuning strategy (`model.py`)
- Training pipeline with augmentation, callbacks, and evaluation (`train.py`)
- Standalone preprocessing script (`preprocess.py`)

---

## 🏗️ Model Architecture

Final solution uses **MobileNetV2** with transfer learning and fine-tuning.

### Design decisions

| Decision | Choice | Reasoning |
|---|---|---|
| Base model | MobileNetV2 | Strong ImageNet features, efficient architecture |
| Fine-tuning | Top layers only (bottom 150 frozen) | Preserve low-level features, adapt high-level |
| Head | Dense(512→256→128→5) | Sufficient capacity for 5-class problem |
| Regularization | L2 + BatchNorm + Dropout | Combat overfitting on small dataset |
| Optimizer | RMSprop + ExponentialDecay | Stable fine-tuning with decaying LR |
| Gradient clipping | global_clipnorm=1.0 | Prevent exploding gradients during fine-tuning |
| Early stopping | patience=6 on val_loss | Prevent overfitting, restore best weights |

### Architectures explored

| Architecture | Notes |
|---|---|
| Custom CNN from scratch | 6 conv blocks, BatchNorm, Dropout |
| ResNet50 + custom head | Transfer learning baseline |
| **MobileNetV2 + fine-tuning** | **Best performance — final solution** |

---

## 🔧 Training Pipeline

- Data augmentation: rotation (±20°), width/height shift, zoom, horizontal flip
- EarlyStopping on validation loss (patience=6, restore best weights)
- Confusion matrix and training history visualization
- Model saved as `.keras` after training

---

## ☁️ Deployment

Model was exported and deployed to **Google Cloud Vertex AI** for 
scalable REST API inference. The deployment pipeline wraps the model 
to accept raw JPEG bytes directly — enabling production prediction calls 
without client-side preprocessing.

Deployment pipeline provided by ML6 challenge template.

---

## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess and inspect the dataset
python preprocess.py

# Train the model
# Note: dataset not included — see ML6 challenge repo for data access
python train.py
```

---

## 📊 Results

- Passed ML6's automated evaluation with high classification accuracy
- Invited to personal interview stage after completing the challenge

---

## 🛠️ Requirements