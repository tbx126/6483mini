# Cat-Dog Image Classification with CNN

A PyTorch-based binary classification project for distinguishing between cat and dog images. This project features a convolutional neural network (CNN) architecture with complete training, validation, and prediction capabilities.

## 📋 Project Overview

This project implements a lightweight CNN model for binary classification of cat and dog images. Key features include:

- **Model Training**: 4-layer convolutional network with batch normalization and dropout
- **Model Validation**: Monitor model performance on validation set
- **Learning Rate Scheduling**: Dynamic learning rate adjustment using ReduceLROnPlateau strategy
- **Single Image Prediction**: Classify a single image
- **Batch Prediction**: Predict on all images in test folder and save results

## 🏗️ Project Structure

```
.
├── cnnClassifier.py      # Model definition, training, and validation logic
├── predict.py            # Prediction and inference script
├── config.py             # Hyperparameters and path configuration
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── dataset/
    ├── train/            # Training dataset
    │   ├── cat/
    │   └── dog/
    ├── val/              # Validation dataset
    │   ├── cat/
    │   └── dog/
    └── test/             # Test dataset
```

## 🛠️ Model Architecture

The model uses the following architecture design:

**Convolutional Layers**:
- Conv Block 1: 3 → 32 channels + BatchNorm + ReLU + MaxPool
- Conv Block 2: 32 → 64 channels + BatchNorm + ReLU + MaxPool
- Conv Block 3: 64 → 128 channels + BatchNorm + ReLU + MaxPool
- Conv Block 4: 128 → 256 channels + BatchNorm + ReLU + MaxPool

**Fully Connected Layers**:
- FC1: 256×14×14 → 512 + Dropout(0.5)
- FC2: 512 → 256 + Dropout(0.5)
- FC3: 256 → 2 (binary classification output)

## ⚙️ Configuration Parameters

Configure the following parameters in `config.py`:

### Data Configuration
- `train_dir`: Training data path (default: `dataset/train`)
- `val_dir`: Validation data path (default: `dataset/val`)
- `test_dir`: Test data path (default: `dataset/test`)
- `model_save_path`: Model save path (default: `best_model.pth`)
- `predictions_save_path`: Predictions save path (default: `predictions.csv`)

### Model Hyperparameters
- `img_size`: Input image size (default: 224×224)
- `num_classes`: Number of classification classes (default: 2)
- `batch_size`: Batch size (default: 32)
- `num_epochs`: Number of training epochs (default: 30)
- `learning_rate`: Learning rate (default: 0.001)
- `weight_decay`: Weight decay (default: 0.0001)
- `dropout_rate`: Dropout ratio (default: 0.5)

### Training Configuration
- `num_workers`: Number of data loading threads (default: 4)
- `pin_memory`: Whether to pin memory (default: True)
- `lr_scheduler_patience`: Learning rate scheduler patience (default: 3)
- `lr_scheduler_factor`: Learning rate decay factor (default: 0.5)

### Class Configuration
- `class_names`: Class names (default: `['cat', 'dog']`)

## 📦 Requirements

Install all dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- Pillow >= 8.0.0
- tqdm >= 4.60.0
- numpy

## 🚀 Usage Guide

### 1. Prepare Data

Organize data according to the following directory structure:

```
dataset/
├── train/
│   ├── cat/
│   │   ├── image1.jpg
│   │   └── ...
│   └── dog/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── cat/
│   └── dog/
└── test/
    ├── image1.jpg
    └── ...
```

### 2. Modify Configuration (Optional)

Edit `config.py` to adjust training parameters and data paths.

### 3. Train Model

```bash
python cnnClassifier.py
```

The training process will:
- Display training and validation loss and accuracy for each epoch
- Automatically save the best performing model as `best_model.pth`
- Show current learning rate adjustments

### 4. Predict Single Image

```bash
python predict.py --image path/to/image.jpg
```

### 5. Batch Prediction

```bash
python predict.py --folder dataset/test
```

Results will be saved as a CSV file containing image names and predictions.

## 📊 Data Augmentation

The following data augmentation strategies are applied during training:

- **Random Resized Crop** (RandomResizedCrop): Scale ratio 0.8-1.0
- **Random Horizontal Flip** (RandomHorizontalFlip): Probability 0.5
- **Random Rotation** (RandomRotation): ±15°
- **Color Jitter** (ColorJitter): Brightness, contrast, saturation, and hue adjustments

Validation set uses standard preprocessing:
- Resize to 256×256
- Center crop to 224×224
- Normalize (ImageNet standard)

## 🎯 Model Optimization

- **Optimizer**: Adam (learning rate: 0.001, weight decay: 0.0001)
- **Loss Function**: CrossEntropyLoss
- **Learning Rate Scheduling**: ReduceLROnPlateau (patience: 3, decay factor: 0.5)
- **Regularization**: Batch Normalization + Dropout

## 💻 Hardware Requirements

- **Recommended**: NVIDIA GPU with CUDA support
- **Minimum**: CPU (slower but functional)

The code automatically detects and uses available GPU; if none, it uses CPU.

## 📝 Output Description

### Training Output
```
Using device: cuda
Training set size: 10000
Validation set size: 2000
Class mapping: {'cat': 0, 'dog': 1}

Starting training...
============================================================
Epoch 1/30 [Train]...
Epoch 1/30 [Val]...
 Current learning rate: 0.001000
 ✓ Best model saved (Val accuracy: 92.50%)
...
```

### Prediction Output
```
Image: cat.jpg → Prediction: cat (Confidence: 95.23%)
Image: dog.jpg → Prediction: dog (Confidence: 98.45%)
```

## 🔍 FAQ

**Q: How do I load the best trained model?**

A: Load the model checkpoint before prediction:
```python
from cnnClassifier import CatDogCNN
import torch

model = CatDogCNN(num_classes=2)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

**Q: Model training is too slow?**

A:
- Ensure GPU is available (use CUDA)
- Increase `batch_size` (requires more VRAM)
- Reduce `num_epochs`

**Q: Validation accuracy is not ideal?**

A:
- Check data quality and label accuracy
- Adjust learning rate or increase training epochs
- Try increasing data augmentation strength
- Increase training dataset size

## 📄 License

MIT License

## ✨ Changelog

- v1.0 (2025-11-12): Initial project release with complete training, validation, and prediction functionality
