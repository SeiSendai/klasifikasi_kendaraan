import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="Vehicle Type Classification", 
    page_icon="🚗", 
    layout="centered"
)

st.title("🚗 Klasifikasi Jenis Kendaraan")
st.write("Aplikasi web berbasis PyTorch untuk mendeteksi kategori kendaraan dari gambar.")

# 1. Definisikan Arsitektur Model Custom CNN
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 2. Fungsi Load Model dengan Cache
@st.cache_resource
def load_model():
    model_path = 'model_custom_cnn.pth'
    if not os.path.exists(model_path):
        return None, None
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    class_names = checkpoint['classes']
    
    model = CustomCNN(len(class_names))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, class_names

model, class_names = load_model()

if model is None:
    st.error("⚠️ File bobot model `model_custom_cnn.pth` tidak ditemukan! Pastikan file model berada di folder yang sama dengan `app.py`.")
else:
    # Sidebar untuk menampilkan daftar kelas
    with st.sidebar:
        st.header("📋 Daftar Kelas Kendaraan")
        st.write("Model ini mengenali kategori berikut:")
        for name in class_names:
            st.code(name)

    # Widget Upload File
    uploaded_file = st.file_uploader("Pilih gambar kendaraan (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.write("---")
        # Membuat 2 kolom rapi: Kiri untuk gambar, Kanan untuk hasil
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.subheader("🖼️ Gambar Masukan")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("📊 Hasil Prediksi")
            
            if st.button("🔍 Prediksi Gambar", type="primary", use_container_width=True):
                with st.spinner("Sedang memproses..."):
                    # Preprocessing
                    transform = transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    
                    input_tensor = transform(image).unsqueeze(0)
                    
                    # Inference
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                        confidence, predicted_idx = torch.max(probabilities, 0)
                    
                    predicted_class = class_names[predicted_idx.item()]
                    confidence_score = confidence.item() * 100
                
                # Tampilkan hasil dengan komponen native streamlit
                st.success(f"**Prediksi Utama:** {predicted_class.upper()}")
                st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence_score:.2f}%")
                
                st.write("---")
                st.write("**Probabilitas Seluruh Kelas:**")
                for idx, prob in enumerate(probabilities):
                    p_val = prob.item()
                    st.text(f"{class_names[idx]}: {p_val * 100:.1f}%")
                    st.progress(p_val)
    else:
        st.info("💡 Silakan unggah gambar kendaraan melalui tombol di atas untuk memulai analisis.")