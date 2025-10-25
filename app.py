import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import joblib

# Load model & scaler
model = joblib.load("models/mnist_best_model.pkl")
scaler = joblib.load("models/mnist_scaler.pkl")

st.title("MNIST Digit Recognition")
st.write("Draw a digit (0-9) below and click Predict")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=15,
    stroke_color="#FFFFFF",  # White pen
    background_color="#000000",  # Black canvas
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert to grayscale 28x28
        img = Image.fromarray(np.uint8(canvas_result.image_data[:, :, 0])).convert("L")
        img = img.resize((28,28))
        img_array = np.array(img).reshape(1, -1)
        img_array = scaler.transform(img_array / 255.0)

        # Predict
        prediction = model.predict(img_array)
        st.success(f"Predicted Digit: {prediction[0]}")
