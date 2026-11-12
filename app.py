import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Car Type Classification AI",
    page_icon="🚗",
    layout="wide"
)

CLASS_NAMES = ['Hatchback', 'Other', 'Pickup', 'Sedan', 'SUV']

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model('model_custom_cnn.keras', compile=False)

try:
    model = load_cnn_model()
except Exception:
    st.error("File model 'model_custom_cnn.keras' tidak ditemukan!")
    st.stop()

st.title("🚗 Car Type Classification — Custom CNN")
st.write("Klasifikasi 5 jenis mobil menggunakan Custom CNN.")

uploaded_file = st.file_uploader("Unggah foto mobil...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, use_container_width=True)
        
    with col2:
        with st.spinner('Memproses citra...'):
            img_resized = image.resize((128, 128))
            img_array = np.array(img_resized, dtype=np.float32)
            
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
                
            img_tensor = np.expand_dims(img_array, axis=0)
            
            preds = model(img_tensor, training=False).numpy()[0]
            
            top_idx = int(np.argmax(preds))
            predicted_label = CLASS_NAMES[top_idx]
            confidence = float(preds[top_idx]) * 100
            
            st.success(f"**Jenis Mobil:** {predicted_label}")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")

    st.markdown("---")
    prob_df = pd.DataFrame({
        'Jenis Mobil': CLASS_NAMES,
        'Probabilitas (%)': np.round(preds * 100, 2)
    }).sort_values(by='Probabilitas (%)', ascending=False)
    
    st.bar_chart(prob_df.set_index('Jenis Mobil'))