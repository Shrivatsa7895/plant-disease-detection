
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -----------------------------------------
# 1. Application configuration
# -----------------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

MODEL_PATH = Path("model/plant_disease_model.keras")
CLASS_PATH = Path("model/class_names.json")

IMAGE_SIZE = (224, 224)

# -----------------------------------------
# 2. Disease information
# -----------------------------------------

DISEASE_INFO = {
    "Tomato_Healthy": {
        "title": "Healthy Tomato Leaf",
        "description": "No target disease was identified in this image.",
        "advice": "Continue regular watering and monitor the plant."
    },
    "Tomato_Early_Blight": {
        "title": "Tomato Early Blight",
        "description": "Early blight is commonly associated with dark spots and concentric rings on older leaves.",
        "advice": "Remove affected leaves, improve airflow and avoid overhead watering."
    },
    "Tomato_Late_Blight": {
        "title": "Tomato Late Blight",
        "description": "Late blight can cause water-soaked lesions that turn brown or black.",
        "advice": "Isolate affected plants and seek appropriate agricultural advice promptly."
    },
    "Potato_Healthy": {
        "title": "Healthy Potato Leaf",
        "description": "No target disease was identified in this image.",
        "advice": "Maintain good crop hygiene and monitor plant health."
    },
    "Potato_Early_Blight": {
        "title": "Potato Early Blight",
        "description": "Early blight can cause brown lesions with characteristic concentric rings.",
        "advice": "Remove infected foliage and maintain good field sanitation."
    }
}


# -----------------------------------------
# 3. Load trained model
# -----------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r") as file:
        class_names = json.load(file)

    if len(class_names) != model.output_shape[-1]:
        raise ValueError(
            "Number of class names does not match model outputs."
        )

    return model, class_names


# -----------------------------------------
# 4. Preprocess uploaded image
# -----------------------------------------

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# -----------------------------------------
# 5. Application interface
# -----------------------------------------

st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a photograph of a tomato or potato leaf "
    "to get an AI-based classification."
)

st.info(
    "This application recognizes five categories: "
    "healthy tomato, tomato early blight, tomato late blight, "
    "healthy potato and potato early blight."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------------------
# 6. Prediction
# -----------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image.verify()

        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Leaf Image")

        st.image(image, caption="Your uploaded leaf photograph", use_column_width=True)

        if st.button(
            "Detect Plant Disease",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Analyzing leaf image..."):

                model, class_names = load_model()

                processed_image = preprocess_image(image)

                predictions = model.predict(
                    processed_image,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(predictions)
                )

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = float(
                    predictions[predicted_index]
                ) * 100

            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Prediction: {DISEASE_INFO.get(predicted_class, {}).get('title', predicted_class)}"
            )

            st.metric(
                "Model confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                float(confidence / 100)
            )

            info = DISEASE_INFO.get(predicted_class, {})

            st.write("### About this result")

            st.write(
                info.get(
                    "description",
                    "No additional information available."
                )
            )

            st.write("### Suggested action")

            st.write(
                info.get(
                    "advice",
                    "Consult a qualified agricultural expert."
                )
            )

            with st.expander("View all class probabilities"):
                for index, name in enumerate(class_names):
                    st.write(
                        f"{name}: {predictions[index] * 100:.2f}%"
                    )

            st.warning(
                "This is an experimental AI classification, "
                "not a confirmed agricultural diagnosis. "
                "Model confidence is not diagnostic certainty. "
                "PlantVillage images may differ from real field "
                "conditions. Consult an agricultural expert "
                "before making treatment decisions."
            )

    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Unable to process this image: {error}")

    except Exception as error:
        st.error(f"An unexpected error occurred: {error}")


# -----------------------------------------
# 7. Footer
# -----------------------------------------

st.divider()

st.caption(
    "Plant Disease Detection | MobileNetV2 | "
    "TensorFlow | Streamlit"
)
