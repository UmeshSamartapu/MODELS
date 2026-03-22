from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import numpy as np

# Dataset folder
dataset_path = "Dataset"

# Image settings - Grayscale is better for signal-based images
img_size = (128, 128)
batch_size = 32
channels = 1 # Changed to 1 for Grayscale

# Data Generator
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=5,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=False,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale", # 🔥 Added grayscale
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale", # 🔥 Added grayscale
    class_mode="binary",
    subset="validation"
)

# Save labels
labels = train_generator.class_indices
labels = {v: k for k, v in labels.items()}
os.makedirs("Model", exist_ok=True)
np.save("Model/labels.npy", labels)

# -------------------------------
# 🔥 IMPROVED CNN MODEL
# -------------------------------
model = Sequential([
    # Input Layer
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Adding a 4th layer for better feature extraction
    Conv2D(256, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation="relu"), # Increased neurons
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# Optimizer with slightly higher start rate
optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint("Model/best_model.h5", monitor="val_accuracy", save_best_only=True)
# 🔥 NEW: Reduces learning rate when accuracy plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100, # Increased to 100
    callbacks=[early_stop, checkpoint, reduce_lr]
)

model.save("Model/model.h5")
print("✅ Improved Model Training Completed")