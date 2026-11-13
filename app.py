import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 1. Konfigurasi Halaman (Harus di baris pertama perintah Streamlit)
st.set_page_config(
    page_title="Vehicle Type Classifier", 
    page_icon="🚗", 
    layout="wide"
)

# Custom CSS untuk memperindah tampilan
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Definisikan Arsitektur Model Custom CNN
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

# 3. Fungsi Load Model dengan Cache
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

# 4. Header Utama Aplikasi
st.markdown('<p class="main-title">🚗 Vehicle Type Classification App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Klasifikasi Jenis Kendaraan Berbasis PyTorch & Custom CNN</p>', unsafe_allow_html=True)

model, class_names = load_model()

if model is None:
    st.error("⚠️ File bobot model `model_custom_cnn.pth` tidak ditemukan di direktori! Pastikan Anda sudah mengunggah atau melatih model terlebih dahulu.")
else:
    # Sidebar Informasi
    with st.sidebar:
        st.header("ℹ️ Informasi Model")
        st.write("Aplikasi ini mendeteksi kategori kendaraan berdasarkan pelatihan model *Custom Convolutional Neural Network*.")
        st.markdown("---")
        st.markdown("**Kelas Kendaraan yang Dikenali:**")
        for name in class_names:
            st.code(name)
        st.markdown("---")
        st.info("💡 **Tips:** Gunakan gambar yang karakteristiknya mirip dengan data latih dataset untuk hasil prediksi optimal.")

    # Layout Utama Menjadi 2 Kolom (Kiri: Upload & Gambar, Kanan: Hasil Prediksi)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📁 Unggah Gambar")
        uploaded_file = st.file_uploader("Pilih gambar kendaraan...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Pratinjau Gambar", use_container_width=True)
        else:
            st.info("Silakan unggah file gambar (JPG/PNG) di atas untuk memulai.")

    with col2:
        st.subheader("📊 Hasil Prediksi")
        
        if uploaded_file is not None:
            if st.button("🔍 Jalankan Prediksi", type="primary", use_container_width=True):
                with st.spinner("Model sedang menganalisis fitur gambar..."):
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
                
                # Tampilkan Hasil di Kotak Khusus
                st.markdown(f"""
                    <div class="result-card">
                        <h3>Prediksi Utama: <span style="color: #2563EB;">{predicted_class.upper()}</span></h3>
                        <p>Tingkat Kepercayaan: <b>{confidence_score:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.markdown("#### Detail Probabilitas per Kategori:")
                
                # Tampilkan Progress Bar untuk setiap kelas
                for idx, prob in enumerate(probabilities):
                    p_val = prob.item()
                    st.text(f"{class_names[idx]} ({p_val * 100:.1f}%)")
                    st.progress(p_val)
        else:
            st.warning("Belum ada gambar yang diunggah. Silakan unggah gambar di sebelah kiri.")