# Goal Segmentation Training Experiment - exp3050ti2

## 📊 Experiment Overview

This experiment trained a **YOLOv8m-seg** model for goal line detection and football segmentation using computer vision techniques. The model was trained for **64 epochs** with advanced segmentation capabilities to detect multiple football-related objects.

## 🎯 Model Configuration

- **Model**: YOLOv8m-seg.pt (segmentation variant)
- **Training Epochs**: 100 (early stopped at 64)
- **Batch Size**: 16
- **Image Size**: 640x640
- **Optimizer**: AdamW
- **Device**: CUDA (GPU 0)
- **Workers**: 8

### Classes Detected
- Ball
- Crossbar
- Line (goal line)
- Person
- Post (goal post)

## 📈 Training Results

### Final Training Metrics (Epoch 64)
| Metric | Box Detection | Mask Segmentation |
|--------|---------------|-------------------|
| **Precision** | 95.32% | 92.64% |
| **Recall** | 90.56% | 87.97% |
| **mAP@0.5** | 94.21% | 90.99% |
| **mAP@0.5:0.95** | 80.58% | 70.15% |

### Loss Values (Final Epoch)
- **Box Loss**: 0.457
- **Segmentation Loss**: 0.689
- **Classification Loss**: 0.391
- **DFL Loss**: 0.887

## 🔥 Training Progress Visualization

### Overall Results
![Training Results](results.png)
*Complete training metrics over all epochs showing loss curves and performance metrics*

### Performance Curves

#### Box Detection Performance
![Box Precision Curve](BoxP_curve.png)
*Precision curve for bounding box detection across all classes*

![Box Recall Curve](BoxR_curve.png)
*Recall curve for bounding box detection across all classes*

![Box F1 Curve](BoxF1_curve.png)
*F1-score curve for bounding box detection showing optimal thresholds*

![Box PR Curve](BoxPR_curve.png)
*Precision-Recall curve for bounding box detection*

#### Mask Segmentation Performance
![Mask Precision Curve](MaskP_curve.png)
*Precision curve for instance segmentation masks*

![Mask Recall Curve](MaskR_curve.png)
*Recall curve for instance segmentation masks*

![Mask F1 Curve](MaskF1_curve.png)
*F1-score curve for instance segmentation*

![Mask PR Curve](MaskPR_curve.png)
*Precision-Recall curve for mask segmentation*

## 🎯 Model Performance Analysis

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)
*Raw confusion matrix showing classification performance*

![Normalized Confusion Matrix](confusion_matrix_normalized.png)
*Normalized confusion matrix for better class performance visualization*

### Class Distribution
![Label Distribution](labels.jpg)
*Distribution of labels in the training dataset*

## 🖼️ Training Samples

### Training Batches
![Training Batch 0](train_batch0.jpg)
*Sample training batch 0 with annotations*

![Training Batch 1](train_batch1.jpg)
*Sample training batch 1 with annotations*

![Training Batch 2](train_batch2.jpg)
*Sample training batch 2 with annotations*

## ✅ Validation Results

### Validation Predictions vs Ground Truth

#### Validation Batch 0
![Validation Batch 0 Labels](val_batch0_labels.jpg)
*Ground truth labels for validation batch 0*

![Validation Batch 0 Predictions](val_batch0_pred.jpg)
*Model predictions for validation batch 0*

#### Validation Batch 1
![Validation Batch 1 Labels](val_batch1_labels.jpg)
*Ground truth labels for validation batch 1*

![Validation Batch 1 Predictions](val_batch1_pred.jpg)
*Model predictions for validation batch 1*

#### Validation Batch 2
![Validation Batch 2 Labels](val_batch2_labels.jpg)
*Ground truth labels for validation batch 2*

![Validation Batch 2 Predictions](val_batch2_pred.jpg)
*Model predictions for validation batch 2*

## ⚙️ Hyperparameters

### Optimization
- **Learning Rate (lr0)**: 0.001
- **Final Learning Rate (lrf)**: 0.01
- **Momentum**: 0.937
- **Weight Decay**: 0.0005
- **Warmup Epochs**: 3.0

### Data Augmentation
- **HSV Augmentation**: H=0.015, S=0.7, V=0.4
- **Translation**: 0.1
- **Scale**: 0.5
- **Horizontal Flip**: 0.5
- **Mosaic**: 1.0
- **Random Augment**: Enabled
- **Erasing**: 0.4

### Loss Weights
- **Box Loss**: 7.5
- **Classification Loss**: 0.5
- **DFL Loss**: 1.5
- **Mask Loss**: Auto-calculated

## 🚀 Key Achievements

1. **Excellent Box Detection**: 94.21% mAP@0.5 for bounding box detection
2. **Strong Segmentation**: 90.99% mAP@0.5 for instance segmentation
3. **High Precision**: 95.32% precision for box detection
4. **Balanced Performance**: Good balance between precision and recall
5. **Early Convergence**: Model converged well before 100 epochs

## 💡 Model Insights

- The model shows excellent performance on all football-related objects
- Segmentation masks provide precise pixel-level detection for goal line analysis
- High precision indicates low false positive rate
- Strong recall ensures most objects are detected
- The validation predictions closely match ground truth labels

## 📁 Files Included

- `results.png` - Complete training visualization
- `results.csv` - Detailed metrics for each epoch
- `confusion_matrix.png` & `confusion_matrix_normalized.png` - Classification analysis
- `*_curve.png` - Performance curves for box and mask detection
- `train_batch*.jpg` - Training data samples
- `val_batch*_*.jpg` - Validation predictions and labels
- `labels.jpg` - Dataset label distribution
- `args.yaml` - Complete training configuration
- `weights/` - Trained model weights

## 🎯 Applications

This trained model enables:
- **Goal Line Detection**: Precise detection of goal lines in football videos
- **Ball Tracking**: Accurate ball segmentation and tracking
- **Post & Crossbar Detection**: Goal structure identification
- **Real-time Analysis**: Fast inference for live video processing
- **Advanced Goal Detection**: Pixel-perfect segmentation for goal line technology

---

*This experiment demonstrates state-of-the-art segmentation performance for football goal detection with 94.21% mAP@0.5 and robust real-world application capabilities.* 