import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load model
model = load_model("Model/model.h5")

# Load labels
labels = np.load("Model/labels.npy", allow_pickle=True).item()

# Load image
img_path = "test.jpg"
img_size = (128, 128)

# 🔥 Changed color_mode to grayscale to match training
img = image.load_img(img_path, target_size=img_size, color_mode="grayscale")
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    predicted_class = 1
else:
    predicted_class = 0

confidence = prediction if prediction > 0.5 else (1 - prediction)

print("\n🔍 FINAL RESULT")
print(f"Predicted Class: {labels[predicted_class]}")
print(f"Confidence: {round(float(confidence) * 100, 2)} %")

print("\n📊 Probabilities:")
print(f"{labels[0]} : {round((1-prediction)*100,2)}%")
print(f"{labels[1]} : {round(prediction*100,2)}%")