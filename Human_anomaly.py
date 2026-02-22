import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os

# ======================
# 1. Dataset path
# ====================== 
dataset_path = r"C:\\Users\\arnav\\OneDrive\\Desktop\\Dev\\BE_Project\\Datasets\\DIAT-RadHAR\\DIAT-RadHAR"  # <-- CHANGE THIS

# ======================
# 2. Parameters
# ======================
img_size = (300, 300)
batch_size = 32
epochs = 25

# ======================
# 3. Load Dataset
# ======================
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.15,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Detected classes:", class_names)

# Prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ======================
# 4. Data Augmentation (No Lambda layers)
# ======================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ======================
# 5. Build Model
# ======================
model = models.Sequential([
    layers.Input(shape=img_size + (3,)),
    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# ======================
# 6. Compile Model
# ======================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# 7. Train Model
# ======================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ======================
# 8. Save Model
# ======================
model.save("micro_doppler_classifier.h5")
print("Model saved as micro_doppler_classifier.h5")
