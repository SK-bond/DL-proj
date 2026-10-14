# Diabetic Retinopathy Grading with Deep Learning

Classifies retinal fundus images into 5 diabetic retinopathy grades using PyTorch:
No DR, Mild, Moderate, Severe, Proliferative DR.

## Models
- ResNet50 (transfer learning)
- VGG16
- DenseNet121
- ViT-B/16

## Dataset
DDR dataset, balanced and split into train/val/test.
The dataset is not included in this repo because of its size.

## Project structure
| File / folder | Purpose |
|---|---|
| data_preparation.py, preprocessing.py | Data loading and preprocessing |
| balance_dataset.py, split_balanced_dataset.py | Class balancing and splitting |
| train_vgg16.py, train_densenet121_balanced.py | Model training |
| evaluate_best_model.py | Test-set evaluation |
| compare_*.py, plot_*.py | Model comparison and plots |
| gradcam.py, gradcam_both_models.py | Grad-CAM explainability |
| predict.py | Predict the DR grade of one image |
| *_results/ | Saved metrics, graphs, and predictions |

## Results
Add the accuracy and F1 comparison table here.

## Explainability
Grad-CAM heatmaps for ResNet50 and VGG16 are in gradcam_both_results/.
