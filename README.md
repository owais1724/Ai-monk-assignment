# Aimonk Multi-Label Image Classification Assignment

This repository contains a professional PyTorch implementation for a multi-label image classification task involving 4 attributes. The system is designed to handle real-world data challenges such as **missing labels (NA values)** and **class imbalance**.

---

## 🚀 Quick Start

### 1. Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install torch torchvision pandas matplotlib scikit-learn tqdm pillow
```

### 2. Training the Model
To start training, simply run:
```bash
python train.py
```
This will:
- Load the images and labels with proper **NA masking**.
- Calculate **class weights** to handle imbalance.
- Fine-tune a pre-trained **ResNet50** model.
- Save the best model to `multilabel_model.pth`.
- Generate the required loss curve: `loss_curve.png`.

### 3. Inference
To predict attributes for a specific image, use the inference script:
```bash
python inference.py --image_path images/image_71.jpg
```

---

## 🧠 Technical Implementation Details

### 1. Architecture & Fine-Tuning
- **Base Model**: ResNet50 (pre-trained on ImageNet-1K).
- **Modification**: The final classification head was replaced with a linear layer mapping to 4 output nodes.
- **Why?**: ResNet50 provides deep feature extraction capabilities, and using pre-trained weights significantly accelerates convergence compared to training from scratch.

### 2. Handling "NA" Labels (The Masking Strategy)
Our requirement was not to ignore images with partial labels. 
- **Implementation**: Instead of removing rows with "NA", we created a **Mask Tensor**.
- **Logic**: During loss calculation, the Binary Cross-Entropy (BCE) loss is multiplied by this mask. If a label is "NA", its mask value is 0, effectively zeroing out the loss for that specific attribute in that specific image. This ensures the model only learns from confirmed data while still utilizing the other valid labels of the same image.

### 3. Handling Class Imbalance
Multi-label datasets are often skewed. 
- **Implementation**: We calculate `pos_weight` for each class (ratio of negative samples to positive samples).
- **Loss Function**: `nn.BCEWithLogitsLoss(pos_weight=weights)` is used. This forces the model to pay more attention to rare attributes, preventing the majority classes from dominating the training process.

---

## 📈 Deliverables
- **Model weights**: `multilabel_model.pth`
- **Loss Plot**: `loss_curve.png`
  - **X-axis**: `iteration_number`
  - **Y-axis**: `training_loss`
  - **Title**: `Aimonk_multilabel_problem`
- **Modular Scripts**: `dataset.py`, `model.py`, `train.py`, `inference.py`, and `utils.py`.

---

## � Pre-processing & Augmentations

### Implemented:
- **Resizing**: Fixed 224x224 input size as expected by ResNet.
- **Normalization**: ImageNet mean and standard deviation.
- **Random Horizontal Flip**: Increases spatial invariance.
- **Random Rotation**: Handles slight tilts and perspective shifts.
- **Color Jitter**: Brightness, contrast, and saturation variations to improve robustness against lighting conditions.

### Proposed Improvements (Time Constraints):
Given more time, the following could be implemented to further boost performance:
1.  **CutMix/Mixup**: Advanced data augmentation techniques for multi-label classification that help the model generalize better between overlapping attributes.
2.  **Focal Loss**: While we used class weights, Focal Loss (gamma=2.0) is often more effective at focusing on "hard" examples.
3.  **TTA (Test-Time Augmentation)**: Running inference on multiple versions of an image (flipped, cropped) and averaging the results for higher accuracy.
4.  **Ensemble Learning**: Combining predictions from different architectures (e.g., EfficientNet + Vision Transformers).
5.  **Multi-Threshold Tuning**: Optimizing the 0.5 threshold for each class independently based on the F1-score on a validation set.
6.  **Progressive Resizing**: Training on smaller images first and then increasing size to capture finer details.

---

## �️ Project Structure
- `dataset.py`: Custom PyTorch Dataset with NA handling logic.
- `model.py`: Model definition and weight loading.
- `train.py`: Main training loop with per-iteration logging.
- `inference.py`: User-friendly prediction script.
- `utils.py`: Masked loss function, metrics, and plotting utilities.
