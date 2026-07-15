# ==========================================================
# BERT EMOTION CLASSIFICATION
# Story 5 - SmartBridge
# ==========================================================

import os
import random
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DataCollatorWithPadding
)

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

print("="*60)
print("BERT EMOTION CLASSIFICATION")
print("="*60)

print()

print("Using Device :", device)

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "student_domain_dataset.csv"
)

LABEL_ENCODER_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "label_encoder.pkl"
)

MODEL_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "models",
    "bert_emotion_model_final"
)

os.makedirs(
    MODEL_OUTPUT,
    exist_ok=True
)

# ==========================================================
# HYPERPARAMETERS
# ==========================================================

MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 80

BATCH_SIZE = 32

LEARNING_RATE = 2e-5

EPOCHS = 3

NUM_CLASSES = 5
# ==========================================================
# LOAD DATASET
# ==========================================================

print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape : {df.shape}")

df = df.drop_duplicates(subset=["text"])
df = df.dropna().reset_index(drop=True)

print(f"After Cleaning : {df.shape}")

# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

print("\n" + "="*60)
print("LOADING LABEL ENCODER")
print("="*60)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

df["label"] = label_encoder.transform(df["target_emotion"])

print("\nEmotion Distribution")

print(df["target_emotion"].value_counts())

print("\nClasses")

for i, emotion in enumerate(label_encoder.classes_):
    print(f"{i} -> {emotion}")

# ==========================================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

print("\n" + "="*60)
print("SPLITTING DATASET")
print("="*60)

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=RANDOM_STATE,
    shuffle=True
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE,
    shuffle=True
)

print(f"Training Samples   : {len(train_df):,}")
print(f"Validation Samples : {len(validation_df):,}")
print(f"Testing Samples    : {len(test_df):,}")

# ==========================================================
# CREATE LISTS
# ==========================================================

train_texts = train_df["text"].tolist()
validation_texts = validation_df["text"].tolist()
test_texts = test_df["text"].tolist()

train_labels = train_df["label"].tolist()
validation_labels = validation_df["label"].tolist()
test_labels = test_df["label"].tolist()

# ==========================================================
# LOAD TOKENIZER
# ==========================================================

print("\n" + "=" * 60)
print("LOADING BERT TOKENIZER")
print("=" * 60)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

print("✔ Tokenizer Loaded Successfully")

# ==========================================================
# TOKENIZATION FUNCTION
# ==========================================================

def tokenize_texts(texts):

    return tokenizer(

        texts,

        truncation=True,

        padding=True,

        max_length=MAX_LENGTH,

        return_tensors="pt"

    )

# ==========================================================
# TOKENIZE DATASETS
# ==========================================================

print("\n" + "=" * 60)
print("TOKENIZING DATASETS")
print("=" * 60)

train_encodings = tokenize_texts(train_texts)

validation_encodings = tokenize_texts(validation_texts)

test_encodings = tokenize_texts(test_texts)

print("✔ Training Tokenization Completed")

print("✔ Validation Tokenization Completed")

print("✔ Testing Tokenization Completed")

print()

print("Training Shape")

print(train_encodings["input_ids"].shape)

print()

print("Validation Shape")

print(validation_encodings["input_ids"].shape)

print()

print("Testing Shape")

print(test_encodings["input_ids"].shape)

# ==========================================================
# CUSTOM DATASET
# ==========================================================

class EmotionDataset(Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        item = {}

        for key, value in self.encodings.items():

            item[key] = value[idx]

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item

# ==========================================================
# CREATE DATASETS
# ==========================================================

print("\n" + "=" * 60)
print("CREATING DATASETS")
print("=" * 60)

train_dataset = EmotionDataset(
    train_encodings,
    train_labels
)

validation_dataset = EmotionDataset(
    validation_encodings,
    validation_labels
)

test_dataset = EmotionDataset(
    test_encodings,
    test_labels
)

print("✔ Training Dataset :", len(train_dataset))
print("✔ Validation Dataset :", len(validation_dataset))
print("✔ Testing Dataset :", len(test_dataset))

# ==========================================================
# DATA COLLATOR
# ==========================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# ==========================================================
# DATALOADERS
# ==========================================================

print("\n" + "=" * 60)
print("CREATING DATALOADERS")
print("=" * 60)

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=data_collator

)

validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=data_collator

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=data_collator

)

print("✔ Train Loader :", len(train_loader))
print("✔ Validation Loader :", len(validation_loader))
print("✔ Test Loader :", len(test_loader))

# ==========================================================
# LOAD BERT MODEL
# ==========================================================

print("\n" + "=" * 60)
print("LOADING BERT MODEL")
print("=" * 60)

model = BertForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=NUM_CLASSES

)

model.to(device)

print("✔ BERT Loaded Successfully")

print()

print(model)

# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE

)

print()

print("✔ Optimizer : AdamW")

print("✔ Learning Rate :", LEARNING_RATE)

# ==========================================================
# TRAINING LOOP
# ==========================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

best_accuracy = 0.0

print("\n" + "="*60)
print("STARTING BERT TRAINING")
print("="*60)

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ==========================
    # TRAIN
    # ==========================

    model.train()

    total_loss = 0

    train_predictions = []
    train_targets = []

    progress_bar = tqdm(train_loader)

    for batch in progress_bar:

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        logits = outputs.logits

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        train_predictions.extend(
            predictions.cpu().numpy()
        )

        train_targets.extend(
            labels.cpu().numpy()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = total_loss / len(train_loader)

    train_accuracy = accuracy_score(
        train_targets,
        train_predictions
    )

    # ==========================
    # VALIDATION
    # ==========================

    model.eval()

    validation_loss = 0

    validation_predictions = []

    validation_targets = []

    with torch.no_grad():

        for batch in validation_loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)

            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask,

                labels=labels

            )

            validation_loss += outputs.loss.item()

            predictions = torch.argmax(

                outputs.logits,

                dim=1

            )

            validation_predictions.extend(

                predictions.cpu().numpy()

            )

            validation_targets.extend(

                labels.cpu().numpy()

            )

    validation_loss /= len(validation_loader)

    validation_accuracy = accuracy_score(

        validation_targets,

        validation_predictions

    )

    precision = precision_score(

        validation_targets,

        validation_predictions,

        average="weighted"

    )

    recall = recall_score(

        validation_targets,

        validation_predictions,

        average="weighted"

    )

    f1 = f1_score(

        validation_targets,

        validation_predictions,

        average="weighted"

    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(validation_loss)

    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(validation_accuracy)

    history["precision"].append(precision)
    history["recall"].append(recall)
    history["f1"].append(f1)

    print()

    print(f"Training Loss      : {train_loss:.4f}")
    print(f"Validation Loss    : {validation_loss:.4f}")

    print(f"Training Accuracy  : {train_accuracy:.4f}")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")

    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1 Score           : {f1:.4f}")

    if validation_accuracy > best_accuracy:

        best_accuracy = validation_accuracy

        print()

        print("Saving Best Model...")

        model.save_pretrained(MODEL_OUTPUT)

        tokenizer.save_pretrained(MODEL_OUTPUT)

print()

print("="*60)
print("TRAINING FINISHED")
print("="*60)

# ==========================================================
# TEST EVALUATION
# ==========================================================

print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

model.eval()

test_predictions = []
test_targets = []

with torch.no_grad():

    for batch in tqdm(test_loader):

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        predictions = torch.argmax(

            outputs.logits,

            dim=1

        )

        test_predictions.extend(

            predictions.cpu().numpy()

        )

        test_targets.extend(

            labels.cpu().numpy()

        )

# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(
    test_targets,
    test_predictions
)

precision = precision_score(
    test_targets,
    test_predictions,
    average="weighted"
)

recall = recall_score(
    test_targets,
    test_predictions,
    average="weighted"
)

f1 = f1_score(
    test_targets,
    test_predictions,
    average="weighted"
)

print()

print(f"Test Accuracy : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(

    test_targets,

    test_predictions,

    target_names=label_encoder.classes_,

    output_dict=True

)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(

    os.path.join(

        MODEL_OUTPUT,

        "classification_report.csv"

    )

)

print()

print("✔ Classification Report Saved")

# ==========================================================
# EVALUATION METRICS
# ==========================================================

metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

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

metrics.to_csv(

    os.path.join(

        MODEL_OUTPUT,

        "evaluation_metrics.csv"

    ),

    index=False

)

print("✔ Evaluation Metrics Saved")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(

    test_targets,

    test_predictions

)

plt.figure(figsize=(8,6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks(

    np.arange(NUM_CLASSES),

    label_encoder.classes_,

    rotation=45

)

plt.yticks(

    np.arange(NUM_CLASSES),

    label_encoder.classes_

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(

            j,

            i,

            cm[i,j],

            ha="center",

            va="center"

        )

plt.tight_layout()

plt.savefig(

    os.path.join(

        MODEL_OUTPUT,

        "confusion_matrix.png"

    ),

    dpi=300

)

plt.close()

print("✔ Confusion Matrix Saved")

# ==========================================================
# TRAINING HISTORY
# ==========================================================

history_df = pd.DataFrame(history)

history_df.to_csv(

    os.path.join(

        MODEL_OUTPUT,

        "training_history.csv"

    ),

    index=False

)

# ==========================================================
# ACCURACY PLOT
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    history["train_accuracy"],

    label="Train"

)

plt.plot(

    history["val_accuracy"],

    label="Validation"

)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(

    os.path.join(

        MODEL_OUTPUT,

        "accuracy_plot.png"

    ),

    dpi=300

)

plt.close()

# ==========================================================
# LOSS PLOT
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    history["train_loss"],

    label="Train"

)

plt.plot(

    history["val_loss"],

    label="Validation"

)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(

    os.path.join(

        MODEL_OUTPUT,

        "loss_plot.png"

    ),

    dpi=300

)

plt.close()

print("✔ Accuracy Plot Saved")
print("✔ Loss Plot Saved")

# ==========================================================
# SAVE HUGGINGFACE MODEL
# ==========================================================

model.save_pretrained(MODEL_OUTPUT)

tokenizer.save_pretrained(MODEL_OUTPUT)

print()

print("✔ HuggingFace Model Saved")

print("✔ config.json")

print("✔ model.safetensors")

print("✔ tokenizer.json")

print("✔ tokenizer_config.json")

print("✔ special_tokens_map.json")

print("✔ vocab.txt")

print()

print("=" * 60)
print("BERT TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)