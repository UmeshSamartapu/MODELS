import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# -----------------------------
# ✅ Load Model
# -----------------------------
model_path = "Model/model.h5"

if not os.path.exists(model_path):
    print("❌ Model file not found!")
    exit()

model = load_model(model_path)
print("✅ Model loaded successfully")

# -----------------------------
# ✅ Load Labels
# -----------------------------
labels_path = "Model/labels.npy"

if not os.path.exists(labels_path):
    print("❌ Labels file not found!")
    exit()

labels = np.load(labels_path, allow_pickle=True).item()
print("✅ Labels:", labels)

# -----------------------------
# ✅ Load Image
# -----------------------------
img_path = "test.jpg"

if not os.path.exists(img_path):
    print("❌ Test image not found!")
    exit()

# 🔥 IMPORTANT FIX (MobileNet requires 224x224)
img_size = (224, 224)

img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

print("✅ Image processed")

# -----------------------------
# ✅ Predict
# -----------------------------
prediction = model.predict(img_array)[0][0]

# Convert to class
predicted_class = 1 if prediction > 0.5 else 0
confidence = prediction if prediction > 0.5 else (1 - prediction)

# -----------------------------
# ✅ Output
# -----------------------------
print("\n🔍 FINAL RESULT")
print("Predicted Class:", labels[predicted_class])
print("Confidence:", round(float(confidence) * 100, 2), "%")

# -----------------------------
# 📊 Probabilities
# -----------------------------
print("\n📊 Probabilities:")
print(f"{labels[0]} : {round((1 - prediction) * 100, 2)}%")
print(f"{labels[1]} : {round(prediction * 100, 2)}%")