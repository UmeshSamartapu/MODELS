import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model('ecg_model.h5')

# Load and Preprocess image
img = Image.open('test.jpg').convert('RGB')
img = img.resize((48, 48))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

# Mapping (Double-check this alphabetical order!)
classes = [
    "Abnormal Heartbeat",
    "History of MI",
    "Myocardial Infarction",
    "Normal"
]
predicted_label = classes[predicted_index]

# --- SAFETY LOGIC ---
THRESHOLD = 70.0  # Require at least 70% confidence

print("-" * 30)
print(f"Raw Prediction Probabilities: {prediction}")

if confidence < THRESHOLD:
    print(f"RESULT: ⚠️ UNCERTAIN (Confidence too low: {confidence:.2f}%)")
    print("Action: Please provide a clearer ECG image.")
else:
    if predicted_label == "Normal":
        print(f"RESULT: Normal ✅ ({confidence:.2f}%)")
    else:
        print(f"RESULT: Arrhythmia Detected ({predicted_label}) ❌ ({confidence:.2f}%)")
print("-" * 30)