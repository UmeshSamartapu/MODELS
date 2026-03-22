import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
dataset_path = "Dataset"
img_size = (224, 224)
batch_size = 16

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model("Model/model.h5")
print("✅ Model loaded")

# -----------------------------
# LOAD LABELS
# -----------------------------
labels = np.load("Model/labels.npy", allow_pickle=True).item()
class_names = list(labels.values())

# -----------------------------
# VALIDATION GENERATOR
# -----------------------------
datagen = ImageDataGenerator(rescale=1.0/255)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False   # 🔥 IMPORTANT
)

# -----------------------------
# GET TRUE LABELS
# -----------------------------
y_true = val_generator.classes

# -----------------------------
# PREDICT
# -----------------------------
y_pred_probs = model.predict(val_generator)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

print("\n📊 Confusion Matrix:")
print(cm)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# -----------------------------
# PLOT CONFUSION MATRIX
# -----------------------------
plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()