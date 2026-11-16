
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ----------------------------------
# 1. Configuration
# ----------------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001
SEED = 42

TRAIN_DIR = Path("dataset/train")
VAL_DIR = Path("dataset/validation")
TEST_DIR = Path("dataset/test")

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"

CLASS_NAMES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Potato_Healthy",
    "Potato_Early_Blight"
]

tf.keras.utils.set_random_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# 2. Check GPU
# ----------------------------------

print("\nChecking available devices...")

gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", gpus)

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        print("GPU memory setting:", error)

# ----------------------------------
# 3. Load datasets
# ----------------------------------

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# ----------------------------------
# 4. Preprocessing
# ----------------------------------

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_images(images, labels):
    images = preprocess_input(
        tf.cast(images, tf.float32)
    )
    return images, labels

train_ds = train_ds.map(
    preprocess_images,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = val_ds.map(
    preprocess_images,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

test_ds = test_ds.map(
    preprocess_images,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# ----------------------------------
# 5. Calculate class weights
# ----------------------------------

class_counts = np.array([1113, 700, 1336, 106, 700])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=np.repeat(
        np.arange(len(CLASS_NAMES)),
        class_counts
    )
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(weights)
}

print("\nClass weights:")
for name, weight in zip(CLASS_NAMES, weights):
    print(f"{name}: {weight:.3f}")

# ----------------------------------
# 6. Load MobileNetV2
# ----------------------------------

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False

# ----------------------------------
# 7. Build classification head
# ----------------------------------

inputs = tf.keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

outputs = Dense(
    len(CLASS_NAMES),
    activation="softmax"
)(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------------
# 8. Training callbacks
# ----------------------------------

callbacks = [
    ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# ----------------------------------
# 9. Train model
# ----------------------------------

print("\nStarting model training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ----------------------------------
# 10. Save class names
# ----------------------------------

with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(CLASS_NAMES, file, indent=4)

# ----------------------------------
# 11. Evaluate test dataset
# ----------------------------------

print("\nEvaluating on test dataset...")

best_model = tf.keras.models.load_model(MODEL_PATH)

test_loss, test_accuracy = best_model.evaluate(test_ds)

print("\nFinal Test Results")
print("------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nModel saved at:", MODEL_PATH)
print("Class names saved successfully!")
print("\nTraining completed!")
