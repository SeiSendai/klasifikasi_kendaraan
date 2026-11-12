import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Vehicle Type Classification", page_icon="🚗", layout="wide")

CLASS_NAMES = ['Bus', 'Car', 'Motorcycle', 'Truck']

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model('model_custom_cnn.keras')

try:
    model = load_cnn_model()
except Exception as e:
    st.error("File 'model_custom_cnn.keras' tidak ditemukan. Pastikan file model berada di folder yang sama dengan app.py.")
    st.stop()

st.title("🚗 Vehicle Type Classification — Custom CNN")
st.write("Aplikasi klasifikasi jenis kendaraan menggunakan arsitektur **Custom Convolutional Neural Network (TensorFlow/Keras)**.")

with st.sidebar:
    st.header("Informasi Model")
    st.info(
        "- **Arsitektur**: Custom 4-Block CNN\n"
        "- **Input Size**: 128 × 128 × 3 (RGB)\n"
        "- **Kelas**: Bus, Car, Motorcycle, Truck"
    )

uploaded_file = st.file_uploader("Unggah gambar kendaraan (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Citra Masukan")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Hasil Prediksi Model")
        with st.spinner("Menganalisis gambar..."):
            img_resized = image.resize((128, 128))
            img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)

            preds = model.predict(img_array, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            predicted_label = CLASS_NAMES[top_idx]
            confidence = float(preds[top_idx]) * 100

            st.success(f"**Jenis Kendaraan:** {predicted_label}")
            st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence:.2f}%")

    st.markdown("---")
    st.subheader("Distribusi Probabilitas Semua Kelas")
    prob_df = pd.DataFrame({
        'Jenis Kendaraan': CLASS_NAMES,
        'Probabilitas (%)': np.round(preds * 100, 2)
    }).sort_values(by='Probabilitas (%)', ascending=False)

    st.bar_chart(prob_df.set_index('Jenis Kendaraan'))