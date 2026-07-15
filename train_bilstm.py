"""
train_bilstm.py

Description
-----------
Train a Bidirectional LSTM model for 5-class emotion classification.

Pipeline
--------
1. Load preprocessing artifacts
2. Compute class weights
3. Build BiLSTM model
4. Train model
5. Evaluate performance
6. Save model and metrics

Author:
SmartBridge Internship Project
"""

import os
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support
)

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

ARTIFACT_DIR = "artifacts"

MODEL_DIR = os.path.join(
    "models",
    "bilstm"
)

os.makedirs(MODEL_DIR, exist_ok=True)

VOCAB_SIZE = 30000
EMBEDDING_DIM = 128
LSTM_UNITS = 128
MAX_SEQUENCE_LENGTH = 80
NUM_CLASSES = 5

BATCH_SIZE = 256
EPOCHS = 15

LEARNING_RATE = 0.001

RANDOM_STATE = 42

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ============================================================
# Utility
# ============================================================

def print_header(title):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# ============================================================
# Load Artifacts
# ============================================================

def load_artifacts():

    print_header("LOADING TRAINING DATA")

    X_train = np.load(
        os.path.join(
            ARTIFACT_DIR,
            "train_sequences.npy"
        )
    )

    X_validation = np.load(
        os.path.join(
            ARTIFACT_DIR,
            "validation_sequences.npy"
        )
    )

    X_test = np.load(
        os.path.join(
            ARTIFACT_DIR,
            "test_sequences.npy"
        )
    )

    y_train = np.load(
        os.path.join(
            ARTIFACT_DIR,
            "train_labels.npy"
        )
    )

    y_validation = np.load(
        os.path.join(
            ARTIFACT_DIR,
            "validation_labels.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            ARTIFACT_DIR,
            "test_labels.npy"
        )
    )

    with open(
        os.path.join(
            ARTIFACT_DIR,
            "label_encoder.pkl"
        ),
        "rb"
    ) as file:

        label_encoder = pickle.load(file)

    print("Training Shape :", X_train.shape)
    print("Validation Shape :", X_validation.shape)
    print("Testing Shape :", X_test.shape)

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        label_encoder
    )
# ============================================================
# Compute Class Weights
# ============================================================

def get_class_weights(y_train):

    print_header("COMPUTING CLASS WEIGHTS")

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = dict(zip(classes, weights))

    print("Computed Class Weights:\n")

    for cls, weight in class_weights.items():
        print(f"Class {cls} : {weight:.4f}")

    return class_weights


# ============================================================
# Build BiLSTM Model
# ============================================================

def build_model():

    print_header("BUILDING BILSTM MODEL")

    model = Sequential([

        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH
        ),

        Bidirectional(

            LSTM(
                LSTM_UNITS,
                dropout=0.3,
                recurrent_dropout=0.3,
                return_sequences=False
            )

        ),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            NUM_CLASSES,
            activation="softmax"
        )

    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(

        optimizer=optimizer,

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )

    model.summary()

    return model
# ============================================================
# Training Callbacks
# ============================================================

def get_callbacks():

    print_header("CONFIGURING CALLBACKS")

    checkpoint_path = os.path.join(
        MODEL_DIR,
        "best_bilstm_model.keras"
    )

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
        min_lr=1e-6,
        verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    print("✔ EarlyStopping")
    print("✔ ReduceLROnPlateau")
    print("✔ ModelCheckpoint")

    return [
        early_stopping,
        reduce_lr,
        model_checkpoint
    ]


# ============================================================
# Train Model
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    class_weights
):

    print_header("TRAINING BILSTM MODEL")

    callbacks = get_callbacks()

    start_time = time.time()

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_validation,
            y_validation
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        class_weight=class_weights,

        verbose=1

    )

    end_time = time.time()

    training_time = end_time - start_time

    print(f"\nTraining Time : {training_time:.2f} seconds")

    return history


# ============================================================
# Save Training History
# ============================================================

def save_history(history):

    print_header("SAVING TRAINING HISTORY")

    history_df = pd.DataFrame(history.history)

    history_path = os.path.join(
        MODEL_DIR,
        "history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print(f"✔ Saved : {history_path}")


# ============================================================
# Plot Accuracy
# ============================================================

def plot_accuracy(history):

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("BiLSTM Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            "accuracy_plot.png"
        )
    )

    plt.close()


# ============================================================
# Plot Loss
# ============================================================

def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("BiLSTM Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            "loss_plot.png"
        )
    )

    plt.close()
# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(model, X_test, y_test, label_encoder):

    print_header("EVALUATING MODEL")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    # --------------------------------------------------------

    predictions = model.predict(
        X_test,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    # --------------------------------------------------------

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predicted_labels,
        average="weighted"
    )

    print("\nOverall Metrics")

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # --------------------------------------------------------

    report = classification_report(
        y_test,
        predicted_labels,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = os.path.join(
        MODEL_DIR,
        "classification_report.csv"
    )

    report_df.to_csv(report_path)

    print(f"\n✔ Classification Report Saved")
    print(report_path)

    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predicted_labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    plt.figure(figsize=(8,6))

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Confusion Matrix")

    confusion_path = os.path.join(
        MODEL_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✔ Confusion Matrix Saved")

    # --------------------------------------------------------

    metrics_df = pd.DataFrame({

        "Metric":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Value":[
            test_accuracy,
            precision,
            recall,
            f1
        ]

    })

    metrics_path = os.path.join(
        MODEL_DIR,
        "evaluation_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print("✔ Evaluation Metrics Saved")

    return predicted_labels

# ============================================================
# Main Function
# ============================================================

def main():

    print_header("BILSTM EMOTION CLASSIFICATION")

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        label_encoder
    ) = load_artifacts()

    class_weights = get_class_weights(y_train)

    model = build_model()

    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        class_weights=class_weights
    )

    save_history(history)

    plot_accuracy(history)

    plot_loss(history)

    print_header("LOADING BEST MODEL")

    best_model_path = os.path.join(
        MODEL_DIR,
        "best_bilstm_model.keras"
    )

    best_model = tf.keras.models.load_model(
        best_model_path
    )

    evaluate_model(
        best_model,
        X_test,
        y_test,
        label_encoder
    )

    print_header("TRAINING COMPLETED")

    print("Model Directory")
    print(MODEL_DIR)

    print("\nGenerated Files")

    generated_files = [

        "best_bilstm_model.keras",
        "history.csv",
        "accuracy_plot.png",
        "loss_plot.png",
        "classification_report.csv",
        "evaluation_metrics.csv",
        "confusion_matrix.png"

    ]

    for file in generated_files:
        print(f"✔ {file}")

    print("\nBiLSTM model training completed successfully!")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()