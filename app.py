import streamlit as st
import numpy as np
from PIL import Image
import pickle

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✏️",
    layout="centered"
)

# Load model
try:
    mlp_model = pickle.load(open("mlp_emnist_model.pkl", "rb"))
except:
    st.error("Model file not found! Train the model first.")
    st.stop()

st.title("✏️ Handwritten Character Recognition")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if st.button("Predict Character"):

    if uploaded_file is None:
        st.warning("Please upload an image")

    else:
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28,28))

        image_array = np.array(image)

        if np.mean(image_array) > 127:
            image_array = 255 - image_array

        image_array = image_array / 255.0

        # Same preprocessing as training
        image_array = np.rot90(image_array, k=3)
        image_array = np.fliplr(image_array)

        image_array = image_array.reshape(1,-1)

        prediction = mlp_model.predict(image_array)[0]

        predicted_letter = chr(prediction + 65)

        st.success(f"Predicted Character: {predicted_letter}")