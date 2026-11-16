# Plant Disease Detection using MobileNetV2

## Project Overview
This project uses deep learning and transfer learning to identify plant diseases from leaf images. It uses MobileNetV2, TensorFlow, and Keras for image classification and Streamlit to provide an interactive web application.

## Technologies Used
- Python
- TensorFlow and Keras
- MobileNetV2
- Streamlit
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

## Supported Classes
The model classifies images into five categories:

1. Tomato Healthy
2. Tomato Early Blight
3. Tomato Late Blight
4. Potato Healthy
5. Potato Early Blight

## Model Performance
- Architecture: MobileNetV2 (Transfer Learning)
- Input image size: 224 × 224
- Test accuracy: 93.18%
- Test images: 851

*Performance is measured on the held-out PlantVillage test split and may differ on real-world field images.*

## Project Structure

```text
Project1/
├── app.py
├── train_model.py
├── prepare_dataset.py
├── evaluate_model.py
├── requirements.txt
├── model/
│   └── class_names.json
└── dataset/
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Project1
```

### 2. Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the PlantVillage dataset and arrange it in the expected directory structure. Then run:

```bash
python prepare_dataset.py
```

### 5. Train the model

```bash
python train_model.py
```

### 6. Launch the application

```bash
streamlit run app.py
```

## Evaluation

Run:

```bash
python evaluate_model.py
```

The evaluation script generates a classification report and confusion matrix.

## Disclaimer
This project is intended for educational and research purposes. Predictions should not be treated as professional agricultural diagnosis.

