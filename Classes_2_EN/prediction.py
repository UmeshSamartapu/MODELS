import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
import os

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "Model/final_weights.h5"
LABELS_PATH = "Model/labels.npy"

# -----------------------------
# LOAD LABELS
# -----------------------------
labels = np.load(LABELS_PATH, allow_pickle=True).item()
print("Labels:", labels)

# -----------------------------
# REBUILD MODEL (SAME AS TRAINING)
# -----------------------------
def build_model():
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # IMPORTANT (no imagenet here)
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
print("✅ Model weights loaded")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0][0]

    print(f"\nRaw Prediction: {prediction:.4f}")

    # Binary classification
    if prediction > 0.5:
        class_id = 1
    else:
        class_id = 0

    class_name = labels[class_id]

    confidence = prediction if class_id == 1 else (1 - prediction)

    print(f"✅ Predicted Class: {class_name}")
    print(f"🔥 Confidence: {confidence*100:.2f}%")

    return class_name, confidence

# -----------------------------
# TEST IMAGE
# -----------------------------
if __name__ == "__main__":
    test_image = "test.jpg"  # change this
    predict_image(test_image)