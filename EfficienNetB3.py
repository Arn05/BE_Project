import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ----------------------------
# 1. Set dataset path
# ----------------------------
DATASET_PATH = r"C:\\Users\\arnav\\OneDrive\\Desktop\\Dev\\BE_Project\\Datasets\\DIAT-RadHAR\\DIAT-RadHAR"   # <-- change this
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 30

# ----------------------------
# 2. Load dataset (train/val split)
# ----------------------------
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.15,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Class names
class_names = train_ds.class_names
print("Detected classes:", class_names)

# ----------------------------
# 3. Prefetch for performance
# ----------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ----------------------------
# 4. Data Augmentation
# ----------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ----------------------------
# 5. Build model using EfficientNetB3
# ----------------------------
base_model = keras.applications.EfficientNetB3(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze base model for transfer learning

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)

# ----------------------------
# 6. Compile model
# ----------------------------
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# 7. Callbacks (save best model)
# ----------------------------
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

earlystop_cb = keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True
)

# ----------------------------
# 8. Train
# ----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# ----------------------------
# 9. Fine-tuning (optional, improves accuracy further)
# ----------------------------
base_model.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # very low LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[checkpoint_cb, earlystop_cb]
)

print("Training finished. Best model saved as best_model.h5")
