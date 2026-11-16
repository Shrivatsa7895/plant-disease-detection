
import random
import shutil
from pathlib import Path

# Source directory containing the original dataset
SOURCE_DIR = Path("dataset/raw/PlantVillage/PlantVillage")

# Destination directory for the prepared dataset
OUTPUT_DIR = Path("dataset")

# Reproducible random split
random.seed(42)

# Map output class names to original folder names
CLASSES = {
    "Tomato_Healthy": "Tomato_healthy",
    "Tomato_Early_Blight": "Tomato_Early_blight",
    "Tomato_Late_Blight": "Tomato_Late_blight",
    "Potato_Healthy": "Potato___healthy",
    "Potato_Early_Blight": "Potato___Early_blight",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SPLITS = ["train", "validation", "test"]

# Check source folder
if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Dataset source folder not found: {SOURCE_DIR}"
    )

# Avoid accidentally overwriting an existing prepared dataset
for split in SPLITS:
    split_dir = OUTPUT_DIR / split
    if split_dir.exists() and any(split_dir.iterdir()):
        raise FileExistsError(
            f"{split_dir} is not empty. "
            "Move or rename the existing prepared dataset first."
        )

# Check that all five source folders exist
for output_class, source_class in CLASSES.items():
    source_path = SOURCE_DIR / source_class

    if not source_path.is_dir():
        raise FileNotFoundError(
            f"Missing class folder: {source_path}"
        )

# Create output folders
for split in SPLITS:
    for class_name in CLASSES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True, exist_ok=True
        )

# Process each class independently
for output_class, source_class in CLASSES.items():

    source_path = SOURCE_DIR / source_class

    images = [
        path for path in source_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(images) < 10:
        raise ValueError(
            f"Not enough images in {source_class}: {len(images)}"
        )

    random.shuffle(images)

    total = len(images)
    train_count = int(total * 0.70)
    validation_count = int(total * 0.15)

    train_images = images[:train_count]
    validation_images = images[
        train_count:train_count + validation_count
    ]
    test_images = images[train_count + validation_count:]

    split_data = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images,
    }

    for split, split_images in split_data.items():

        destination = OUTPUT_DIR / split / output_class

        for image_path in split_images:
            shutil.copy2(
                image_path,
                destination / image_path.name
            )

    print(f"\n{output_class}")
    print(f"  Total:      {total}")
    print(f"  Training:   {len(train_images)}")
    print(f"  Validation: {len(validation_images)}")
    print(f"  Testing:    {len(test_images)}")

print("\nDataset preparation completed successfully!")
