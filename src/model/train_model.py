import numpy as np
import pandas as pd
from keras import models, layers
from sklearn.metrics import roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from icecream import ic
import os

# Load data
for data_label in ['X_train_train', 'X_val', 'y_train_train', 'y_val']:
    df = np.load(f"data/ml_ready/{data_label}.npy")
    ic(data_label, df.shape)
    globals()[data_label] = df

# Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Single neuron for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])


# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_train), y=y_train_train)
class_weights_dict = dict(enumerate(class_weights))

# MLflow tracking
with mlflow.start_run():
    mlflow.keras.autolog()

    # Train the model with class weights
    history = model.fit(X_train_train, y_train_train, epochs=5, validation_data=(X_val, y_val), class_weight=class_weights_dict)

    # Calculate ROC AUC for train and validation
    y_train_pred = model.predict(X_train_train).ravel()
    y_val_pred = model.predict(X_val).ravel()
    fpr_train, tpr_train, _ = roc_curve(y_train_train, y_train_pred)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_val = auc(fpr_val, tpr_val)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train:.2f})')
    plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label=f'Validation ROC curve (area = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Create visualization folder
    vis_folder = 'mlflow_artifacts/visualizations'
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    # Save ROC curve in the folder
    roc_curve_path = os.path.join(vis_folder, 'roc_curve.png')
    plt.savefig(roc_curve_path)

    # Log metrics and artifacts
    mlflow.log_metric("auc_train", roc_auc_train)
    mlflow.log_metric("auc_val", roc_auc_val)
    mlflow.log_artifacts(vis_folder, artifact_path='visualizations')

    # End the MLflow run
    mlflow.end_run()