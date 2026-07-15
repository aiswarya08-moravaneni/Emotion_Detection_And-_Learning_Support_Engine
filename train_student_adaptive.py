import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ARTIFACT_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "student_adaptive"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "bilstm"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "attention_bilstm_model.keras"
)

OUTPUT_MODEL = os.path.join(
    MODEL_DIR,
    "bilstm_student_adaptive.keras"
)

LABEL_ENCODER_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "label_encoder.pkl"
)

# ==========================================================
# LOAD DATA
# ==========================================================

print("="*60)
print("DOMAIN ADAPTIVE FINE TUNING")
print("="*60)

X_train = np.load(
    os.path.join(
        ARTIFACT_DIR,
        "student_train_sequences.npy"
    )
)

y_train = np.load(
    os.path.join(
        ARTIFACT_DIR,
        "student_train_labels.npy"
    )
)

X_validation = np.load(
    os.path.join(
        ARTIFACT_DIR,
        "student_validation_sequences.npy"
    )
)

y_validation = np.load(
    os.path.join(
        ARTIFACT_DIR,
        "student_validation_labels.npy"
    )
)

print()

print("Training Shape :", X_train.shape)
print("Validation Shape :", X_validation.shape)

print()

print("="*60)
print("LOADING MODEL")
print("="*60)

model = load_model(MODEL_PATH)

print(model.summary())

with open(LABEL_ENCODER_PATH,"rb") as f:
    label_encoder = pickle.load(f)

# ==========================================================
# FREEZE EMBEDDING LAYER
# ==========================================================

print()
print("=" * 60)
print("FREEZING EMBEDDING LAYER")
print("=" * 60)

# Freeze only Embedding Layer
model.layers[0].trainable = False

print("Embedding Layer Frozen")

print()

print("Trainable Layers:")

for layer in model.layers:

    print(
        f"{layer.name:25} Trainable : {layer.trainable}"
    )

# ==========================================================
# COMPILE MODEL
# ==========================================================

print()
print("=" * 60)
print("COMPILING MODEL")
print("=" * 60)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4
)

model.compile(

    optimizer=optimizer,

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

print("Model Compiled Successfully")

# ==========================================================
# CALLBACKS
# ==========================================================

print()
print("=" * 60)
print("CONFIGURING CALLBACKS")
print("=" * 60)

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True,

    verbose=1

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    verbose=1

)

checkpoint = ModelCheckpoint(

    OUTPUT_MODEL,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

callbacks = [

    early_stopping,

    reduce_lr,

    checkpoint

]

print("✔ EarlyStopping")
print("✔ ReduceLROnPlateau")
print("✔ ModelCheckpoint")
# ==========================================================
# TRAIN MODEL
# ==========================================================

print()
print("=" * 60)
print("DOMAIN ADAPTIVE TRAINING")
print("=" * 60)

history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_validation,

        y_validation

    ),

    epochs=8,

    batch_size=64,

    callbacks=callbacks,

    verbose=1,

    shuffle=True

)

print()

print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

# ==========================================================
# SAVE HISTORY
# ==========================================================

history_df = pd.DataFrame(history.history)

history_path = os.path.join(

    MODEL_DIR,

    "student_adaptive_history.csv"

)

history_df.to_csv(

    history_path,

    index=False

)

print(f"✔ Saved : {history_path}")

# ==========================================================
# ACCURACY PLOT
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"

)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"

)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Student Adaptive Training Accuracy")

plt.legend()

accuracy_plot = os.path.join(

    MODEL_DIR,

    "student_adaptive_accuracy.png"

)

plt.savefig(

    accuracy_plot,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ==========================================================
# LOSS PLOT
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    history.history["loss"],

    label="Training Loss"

)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"

)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Student Adaptive Training Loss")

plt.legend()

loss_plot = os.path.join(

    MODEL_DIR,

    "student_adaptive_loss.png"

)

plt.savefig(

    loss_plot,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("✔ Accuracy Plot Saved")

print("✔ Loss Plot Saved")
# ==========================================================
# LOAD BEST MODEL
# ==========================================================

print()
print("=" * 60)
print("LOADING BEST ADAPTIVE MODEL")
print("=" * 60)

best_model = load_model(OUTPUT_MODEL)

# ==========================================================
# EVALUATE MODEL
# ==========================================================

print()
print("=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

loss, accuracy = best_model.evaluate(
    X_validation,
    y_validation,
    verbose=0
)

print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy:.4f}")

# ==========================================================
# PREDICTIONS
# ==========================================================

y_probability = best_model.predict(
    X_validation,
    verbose=0
)

y_prediction = np.argmax(
    y_probability,
    axis=1
)

# ==========================================================
# METRICS
# ==========================================================

precision = precision_score(
    y_validation,
    y_prediction,
    average="weighted"
)

recall = recall_score(
    y_validation,
    y_prediction,
    average="weighted"
)

f1 = f1_score(
    y_validation,
    y_prediction,
    average="weighted"
)

print()
print("Overall Metrics")

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    y_validation,
    y_prediction,
    target_names=label_encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_path = os.path.join(
    MODEL_DIR,
    "student_adaptive_classification_report.csv"
)

report_df.to_csv(
    report_path,
    index=True
)

print()
print("✔ Classification Report Saved")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_validation,
    y_prediction
)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

cm_path = os.path.join(
    MODEL_DIR,
    "student_adaptive_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path,
    index=True
)

print("✔ Confusion Matrix Saved")

# ==========================================================
# EVALUATION METRICS
# ==========================================================

metrics_df = pd.DataFrame({

    "Metric":[
        "Validation Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1
    ]

})

metrics_path = os.path.join(
    MODEL_DIR,
    "student_adaptive_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

print("✔ Metrics Saved")

# ==========================================================
# SUMMARY
# ==========================================================

print()
print("=" * 60)
print("DOMAIN ADAPTIVE TRAINING COMPLETED")
print("=" * 60)

print()

print("Generated Files")

print("✔ bilstm_student_adaptive.keras")
print("✔ student_adaptive_history.csv")
print("✔ student_adaptive_accuracy.png")
print("✔ student_adaptive_loss.png")
print("✔ student_adaptive_classification_report.csv")
print("✔ student_adaptive_confusion_matrix.csv")
print("✔ student_adaptive_metrics.csv")

print()

print("=" * 60)
print("EPIC 2 - STORY 4 COMPLETED SUCCESSFULLY")
print("=" * 60)