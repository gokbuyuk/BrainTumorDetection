# Brain Tumor Detection Project

## Project Overview

This project is currently a **Work in Progress**. It aims to leverage the power of Convolutional Neural Networks (CNNs) to classify brain tumors. The focus is on two main aspects:

1. **Binary Classification**: Identifying whether a brain tumor is present or not.
2. **Multi-class Classification**: Classifying the type of brain tumor such as glioma, meningioma, or pituitary tumor.

## Tools and Methods

### Data

The data for this project consists of brain MRI images. These images are preprocessed and labeled for training two different CNN models. The data is stored and handled in CSV files, which include image paths, labels, and image array data.
Data source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Convolutional Neural Networks (CNN)

We employ CNNs due to their proven effectiveness in image recognition and classification tasks. CNNs can automatically and adaptively learn spatial hierarchies of features from images, making them ideal for our project.

### Binary Classification Model

This model focuses on distinguishing between the presence and absence of brain tumors. It is a fundamental step in automated medical diagnosis, aiding in early detection and treatment planning.

### Multi-class Classification Model

For more detailed analysis, the multi-class model categorizes brain tumors into specific types. This helps in understanding the tumor's nature and deciding on the appropriate medical intervention.

### MLflow for Experiment Tracking

We use MLflow for tracking experiments, managing the machine learning lifecycle, and storing model artifacts. MLflow offers a seamless way to log metrics, parameters, and models during the training process.

### Preprocessing and Data Augmentation

The preprocessing steps include resizing images, normalizing pixel values, and converting image data into NumPy arrays for efficient processing. Data augmentation techniques are also employed to improve the robustness and accuracy of the models.

## Future Work

Upon completion, the project will provide a comprehensive tool for automated brain tumor detection and classification. Further improvements may include integrating advanced neural network architectures and exploring additional data augmentation strategies.
