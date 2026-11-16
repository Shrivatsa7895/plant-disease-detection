import tensorflow as tf
import numpy as np
import streamlit as st
import cv2
import pandas as pd

print("Environment setup successful!")

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("Streamlit:", st.__version__)
print("OpenCV:", cv2.__version__)

print("Available devices:")
print(tf.config.list_physical_devices())

print("All libraries imported successfully!")


print("TensorFlow version:", tf.__version__)

print("Available devices:")
print(tf.config.list_physical_devices())

print("Available GPUs:")
print(tf.config.list_physical_devices("GPU"))