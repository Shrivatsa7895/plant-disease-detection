
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Paths and settings
TEST_DIR = "dataset/test"
MODEL_PATH = "model/plant_disease_model.keras"
CLASS_PATH = "model/class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# Load class names in the exact training order
with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

print("Classes:", class_names)


# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


# Prepare test images
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_names,
    class_mode="sparse",
    shuffle=False
)


# Evaluate model
test_loss, test_accuracy = model.evaluate(test_data, verbose=1)

print("\nTEST RESULTS")
print("---------------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Generate predictions
test_data.reset()
predictions = model.predict(test_data, verbose=1)

predicted_labels = np.argmax(predictions, axis=1)
true_labels = test_data.classes


# Classification report
print("\nCLASSIFICATION REPORT")
print("---------------------------")

report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
)

print(report)

Path("model").mkdir(exist_ok=True)
with open("model/classification_report.txt", "w") as file:
    file.write(report)


# Confusion matrix
cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names)))
)

print("\nCONFUSION MATRIX")
print("---------------------------")
print(cm)


# Plot and save confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(11, 9))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("Plant Disease Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig("model/confusion_matrix.png", dpi=300)
plt.show()

print("\nEvaluation completed!")
print("Saved classification report: model/classification_report.txt")
print("Saved confusion matrix: model/confusion_matrix.png")
