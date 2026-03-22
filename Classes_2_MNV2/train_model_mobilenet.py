import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# CONFIG
# -----------------------------
dataset_path = "Dataset"
img_size = (224, 224)   # 🔥 IMPORTANT
batch_size = 16

# -----------------------------
# DATA GENERATOR
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# Save labels
labels = train_generator.class_indices
labels = {v: k for k, v in labels.items()}
os.makedirs("Model", exist_ok=True)
np.save("Model/labels.npy", labels)
print("Labels:", labels)

# -----------------------------
# 🔥 MobileNetV2 BASE MODEL
# -----------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

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

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "Model/best_model.h5",
    monitor="val_accuracy",
    save_best_only=True
)

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save("Model/model.h5")

print("✅ MobileNet Model Training Completed!")