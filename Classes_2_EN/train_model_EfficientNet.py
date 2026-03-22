import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# CONFIG
# -----------------------------
dataset_path = "Dataset"
img_size = (224, 224)
batch_size = 16

# -----------------------------
# DATA GENERATOR
# -----------------------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# -----------------------------
# SAVE LABELS
# -----------------------------
labels = train_generator.class_indices
labels = {v: k for k, v in labels.items()}
os.makedirs("Model", exist_ok=True)
np.save("Model/labels.npy", labels)

# -----------------------------
# EFFICIENTNET BASE
# -----------------------------
base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# -----------------------------
# CUSTOM HEAD
# -----------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# CALLBACKS
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Saving as .h5 weights only to avoid JSON error during training
checkpoint = ModelCheckpoint(
    "Model/best_weights.h5", 
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# -----------------------------
# TRAIN
# -----------------------------
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# -----------------------------
# FINE-TUNING
# -----------------------------
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("🔄 Fine-tuning...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    verbose=1
)

# -----------------------------
# SAVE FINAL MODEL (FIXED FOR PYTHON 3.7 / TF 2.x)
# -----------------------------
# -----------------------------
# SAVE FINAL MODEL (FIXED)
# -----------------------------

os.makedirs("Model", exist_ok=True)

try:
    # ✅ Best method for your setup
    model.save("Model/final_model.h5")
    print("✅ Model saved successfully as final_model.h5")

except Exception as e:
    print(f"❌ H5 save failed: {e}")

    # 🔥 Ultimate fallback (100% safe)
    model.save_weights("Model/final_weights.h5")
    print("✅ Saved weights only as fallback")