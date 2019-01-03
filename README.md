# Deep Keypoint Detection for the Aesthetic Evaluation of Breast Cancer Surgery Outcomes
Implementation related to the paper: Deep Keypoint Detection for The Aesthetic Evaluation of Breast Cancer Surgery Outcomes.

## Installation 
To run the scripts, you need to have installed: 
* Python 3.5.2 
* Tensorflow 1.2.0
* Keras 2.0.8
* Skimage 0.13.1
* OpenCV 3.4.0

## Instructions 

### Deep Model

Pre-processing steps:

1. Convert images and keypoints to the same size
2. Generation of the heatmaps

Training: 

1. Image Data Generator for Data Augmentation - train_generator.py
2. Main File - train_cv.py

## Citation
If you use this repository, please cite this paper:

> Wilson Silva, Eduardo Castro, Maria J. Cardoso, Florian Fitzal, Jaime S. Cardoso (2019). Deep Keypoint Detection for the Aesthetic Evaluation of Breast Cancer Surgery Outcomes. In Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI'19)

## Datasets (PORTO, TSIO, and VIENNA): 
To have access to the datasets, please contact the first author of the paper. E-mail: wilsonjsilva@inesctec.pt
