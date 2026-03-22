import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "Dataset"
MODEL_PATH = "Model/final_weights.h5"
LABELS_PATH = "Model/labels.npy"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# -----------------------------
# LOAD LABELS (FIXED)
# -----------------------------
labels = np.load(LABELS_PATH, allow_pickle=True).item()

# labels example: {0: 'Normal', 1: 'Arrhythmia'}
class_names = [labels[i] for i in sorted(labels.keys())]

print("Classes:", class_names)

# -----------------------------
# DATA GENERATOR
# -----------------------------
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False   # IMPORTANT
)

# -----------------------------
# BUILD MODEL
# -----------------------------
def build_model():
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

model = build_model()

# -----------------------------
# LOAD WEIGHTS
# -----------------------------
model.load_weights(MODEL_PATH)
print("✅ Weights loaded successfully")

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# EVALUATE
# -----------------------------
loss, accuracy = model.evaluate(val_generator)

print(f"\n📊 Loss: {loss:.4f}")
print(f"🎯 Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# PREDICTIONS
# -----------------------------
predictions = model.predict(val_generator)

# Convert probabilities → binary labels
y_pred = (predictions > 0.5).astype(int).reshape(-1)
y_true = val_generator.classes

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
print("\n🧩 Confusion Matrix:")
print(cm)

# -----------------------------
# CLASSIFICATION REPORT (FIXED)
# -----------------------------
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# -----------------------------
# ROC-AUC SCORE (BONUS)
# -----------------------------
auc = roc_auc_score(y_true, predictions)
print(f"\n🔥 ROC-AUC Score: {auc:.4f}")