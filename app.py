import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_custom_cnn.keras")

model = load_model()
IMAGE_SIZE = 128 
CLASSES = ['Hatchback', 'Other', 'Pickup', 'Sedan', 'SUV']

st.title("Klasifikasi Citra Custom CNN (.keras)")
st.write("Unggah gambar kendaraan untuk diprediksi oleh model.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    # Preprocessing
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Prediksi
    predictions = model.predict(x)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100

    st.subheader(f"Hasil Prediksi: {CLASSES[predicted_index]}")
    st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")