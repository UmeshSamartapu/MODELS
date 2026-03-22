import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('ecg_model.h5')

print("--- MODEL INSPECTION ---")

# 1. Verify Input Shape
input_shape = model.layers[0].input_shape
print(f"Expected Input Shape: {input_shape}")

# 2. Verify Output (Number of Classes)
output_shape = model.layers[-1].output_shape
print(f"Number of Output Classes: {output_shape[1]}")

# 3. Check for Class Names (if saved in the model)
if hasattr(model, 'class_names'):
    print(f"Stored Class Names: {model.class_names}")
else:
    print("Class names not stored in model metadata. Manual verification needed.")

# 4. Model Summary (To see the architecture)
model.summary()