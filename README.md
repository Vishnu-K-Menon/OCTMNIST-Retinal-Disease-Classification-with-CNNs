# OCTMNIST Retinal Disease Classification with CNNs

This repository contains a full deep learning pipeline for classifying retinal OCT (Optical Coherence Tomography) images from the **OCTMNIST** dataset using convolutional neural networks (CNNs).  

The project explores:
- Class imbalance handling with **WeightedRandomSampler**
- Baseline CNN with **CrossEntropyLoss** and **NLLLoss**
- Architectural and training improvements using **Batch Normalization**, **Data Augmentation**, **LR Scheduling**, and **Early Stopping**
- Detailed evaluation via **accuracy/loss curves**, **confusion matrix**, and **ROC curves**

---

## 🧬 Dataset Overview

We use the **OCTMNIST** dataset from MedMNIST:

- **Training Set:** 97,477 images  
- **Test Set:** 1,000 images  
- **Image Size:** 28 × 28 (grayscale, preprocessed by MedMNIST)  
- **Classes:** 4 (labeled `0`, `1`, `2`, `3` – representing different retinal conditions / normal retina)

Each image is a 2D retinal cross-section obtained via OCT.

---

## 📊 Exploratory Data Analysis

### 1. Dataset Visualization

**Per-Class Image Grid**

Random samples from each class are displayed in a grid (one row per class) to visually inspect:

- Structural/textural differences between classes  
- That data loading and transforms are correct  

> 📁 Suggested: save as `assets/class_examples.png` and embed here.

---

### 2. Class Distribution (Bar Graph)

A bar plot of training set class frequencies shows:

- **Class 3** has the highest number of samples  
- Followed by **Class 0**  
- **Classes 1 and 2** are clearly underrepresented

This reveals a **class imbalance** that can bias the model if not addressed.

> ✔ We handle this using a **WeightedRandomSampler**, so minority classes are sampled more frequently during training.

---

### 3. Correlation Matrix of Row-Averaged Intensities

A correlation matrix of **row-averaged pixel intensities** shows:

- Strong positive correlations between **adjacent rows**
- Weaker or even negative correlations between **distant rows**

This suggests local regions share similar structural patterns, which supports the use of CNNs to capture local features.

---

### 4. Scatter Plot: Mean vs Standard Deviation of Intensities

A scatter plot of **mean vs. standard deviation** of pixel intensities (color-coded by class) reveals:

- A **positive correlation**: images with higher mean intensity tend to have higher variation  
- Overlap between classes, but hints of clustering in this feature space

---

## ⚖️ Handling Class Imbalance

To address the imbalance across the four classes, we use **WeightedRandomSampler** in PyTorch:

- Compute class frequencies from the training labels  
- Assign **higher sampling weights** to minority classes  
- Sample **with replacement** so each epoch is effectively balanced  

This exposes the model to minority-class examples more often and helps it learn fairer decision boundaries.

Implementation is in `src/dataset.py` → `create_weighted_sampler()` and `get_dataloaders()`.

---

## 🧠 Baseline CNN Architecture

The core CNN (implemented in `src/models.py`) is:

1. `Conv2d(1 → 32)` + ReLU  
2. `Conv2d(32 → 64)` + ReLU  
3. `MaxPool2d(kernel_size=2, stride=2)`  
4. `Conv2d(64 → 128)` + ReLU  
5. `MaxPool2d(kernel_size=2, stride=2)`  
6. `Dropout(p=0.25)`  
7. `Linear(128 × 7 × 7 → 128)` + ReLU  
8. `Dropout(p=0.25)`  
9. `Linear(128 → 4)` (class logits)

**Training configuration:**

- **Input:** 28×28 grayscale OCT images  
- **Output:** Logits for 4 classes (0, 1, 2, 3)  
- **Loss:** Cross-Entropy or NLLLoss (depending on experiment)  
- **Optimizer:** Adam, `lr = 0.001`  
- **Batch Size:** 64  
- **Epochs:** up to 25 (with early stopping in some configs)  

---

## 📈 Baseline Performance

After 25 epochs with a WeightedRandomSampler, one baseline CNN run achieved approximately:

- **Training Accuracy:** ~93.8%  
- **Validation Accuracy:** ~90.9%  
- **Test Accuracy:** ~75.7%  
- **Training Loss:** ↓ to ~0.16  
- **Validation Loss:** ~0.29  
- **Test Loss:** ~0.87  

### Accuracy & Loss Curves

- Training & validation accuracy increase steadily, with validation accuracy plateauing around 90%.
- Train and validation loss both decrease; the train loss ends lower than the validation loss, indicating **mild overfitting** (expected for a strong CNN on a small test set).

> 📁 Suggested plots:
> - `assets/accuracy_loss_base.png`  
> - `assets/accuracy_loss_model_c.png`  

---

## 🔍 Confusion Matrix & ROC

### Confusion Matrix

On the test set:

- **Classes 0 and 1**: good diagonal dominance → strong performance  
- **Class 2**: more confusion, often misclassified as **Class 0** or **Class 3**, consistent with fewer training examples  
- **Class 3**: strong performance but some misclassification into classes 1 and 2  

This reinforces the impact of class imbalance on minority classes.

---

### ROC Curves (One-vs-All)

One-vs-all ROC analysis gives:

- **Class 0:** AUC ≈ 0.96  
- **Class 1:** AUC ≈ 0.95  
- **Class 2:** AUC ≈ 0.87  
- **Class 3:** AUC ≈ 0.96  

Classes 0, 1, and 3 show excellent separability, while **Class 2** remains more challenging, though still far above random chance.

> 📁 Suggested plot: `assets/roc_curves_base.png` and `assets/roc_curves_model_c.png`

---

## 🧪 Loss Function Experiments: CrossEntropy vs NLL

To investigate loss functions, we trained two setups:

1. **CrossEntropyLoss Model**
2. **NLLLoss Model** (with `F.log_softmax` at the final layer)

Shared config:

- Optimizer: Adam, `lr=0.001`  
- Batch Size: 64  
- Epochs: 20  
- WeightedRandomSampler for imbalance  

### Accuracy Trends

**CrossEntropyLoss model:**

- Training accuracy: starts ~61.8%, improves to >91%  
- Validation accuracy: peaks around **88–90%**  
- Test accuracy: stabilizes around **76–78%**

**NLLLoss model:**

- Training accuracy: surpasses **93%** by later epochs  
- Validation accuracy: also ~88–90%  
- Test accuracy: occasionally reaches **81–82%**, later settling in **76–79%**

Both losses give **similar performance**, with NLLLoss showing slightly stronger peaks in test performance in some runs.

---

### Loss Trends

- Both **CE** and **NLL** quickly reduce training loss, converging near **0.18–0.22** by epoch 20.  
- Validation loss stabilizes around **0.29–0.34** for both models.  
- Test loss oscillates between **0.60–0.80**, hinting at mild overfitting and test-set variance.

Overall:

- CrossEntropy may converge a bit faster initially.
- NLLLoss can match or slightly outperform CE in later epochs, depending on initialization and training dynamics.

---

## 🧪 Advanced Models: A, B, C

To further improve performance, three model configurations were explored, all using **NLLLoss**:

### 🅰 Model A – BatchNorm-only + NLLLoss

- **Architecture:** CNN with **BatchNorm2d** after each convolutional layer  
- **No data augmentation**  
- Still uses WeightedRandomSampler for imbalance  

### 🅱 Model B – DataAug-only + NLLLoss

- No BatchNorm  
- Training-time augmentations:
  - `RandomRotation`
  - `RandomHorizontalFlip`
- WeightedRandomSampler enabled  

### 🅲 Model C – BN + Aug + LR Scheduler + Early Stopping (Best Model)

- Combines:
  - Batch Normalization in all conv layers  
  - Data Augmentation (rotation + horizontal flip)  
  - **ReduceLROnPlateau** learning rate scheduler  
  - **Early stopping** when validation loss does not improve for 8 epochs  

This is the **best-performing model**.

---

## 🏆 Final Performance – Best Model (C)

After training up to 15–20 epochs:

- **Model A (BN-only):** Test Accuracy ≈ **80.50%**
- **Model B (DataAug-only):** Test Accuracy ≈ **79.50%**
- **Model C (BN + Aug + Scheduler + ES):** Test Accuracy ≈ **82.10%**

Detailed metrics for **Model C**:

- **Test Accuracy:** 0.8210  
- **Test Loss:** 0.4993  
- **Precision:** 0.8296  
- **Recall:** 0.8210  
- **F1 Score:** 0.8132  

Model C shows:

- Lowest validation and test losses  
- Most stable accuracy curves  
- Best balance between underfitting and overfitting  

Hence, **Model C** is selected as the final model.

---

## 🧪 Running the Code

### 1. Install Dependencies

```bash
pip install -r requirements.txt


2. Train a Model

Best model (Model C – BN + Aug + Scheduler + ES):
cd src
python train.py --model c --epochs 20
