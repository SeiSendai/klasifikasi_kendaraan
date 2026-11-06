# ==========================================
# FILE: app.py
# ==========================================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd

# 1. Definisi Arsitektur Custom CNN
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

# 2. Fungsi Load Model & Daftar Kelas
@st.cache_resource
def load_model_and_classes():
    checkpoint = torch.load('model_custom_cnn.pth', map_location=torch.device('cpu'))
    class_names = checkpoint['classes']
    model = CustomCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, class_names

# Validasi ketersediaan file model
try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error("File model 'model_custom_cnn.pth' tidak ditemukan. Harap letakkan file model hasil training di direktori yang sama dengan aplikasi ini.")
    st.stop()

# 3. Konfigurasi Transformasi Citra Uji
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Antarmuka Pengguna (UI) Menggunakan Streamlit
st.title("🚗 Vehicle Type Classification")
st.write("Aplikasi web untuk menguji model **Custom CNN** pada klasifikasi jenis kendaraan.")

st.sidebar.header("Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini dibangun menggunakan PyTorch dan Streamlit. "
    "Unggah gambar kendaraan berformat JPG atau PNG untuk melihat hasil prediksi kelas serta tingkat probabilitasnya."
)

uploaded_file = st.file_uploader("Pilih gambar kendaraan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Terunggah")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Hasil Analisis Model")
        with st.spinner('Memproses gambar...'):
            # Preprocessing citra masukan
            img_tensor = transform(image).unsqueeze(0)
            
            # Prediksi menggunakan model Custom CNN
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top_prob, top_class = torch.max(probabilities, 0)
                
            predicted_label = class_names[top_class.item()]
            confidence = top_prob.item() * 100
            
            st.success(f"**Prediksi Jenis:** {predicted_label}")
            st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence:.2f}%")

    # Visualisasi Probabilitas Seluruh Kelas
    st.markdown("---")
    st.subheader("Distribusi Probabilitas Semua Kelas")
    
    prob_df = pd.DataFrame({
        'Jenis Kendaraan': class_names,
        'Probabilitas (%)': (probabilities.numpy() * 100).round(2)
    }).sort_values(by='Probabilitas (%)', ascending=False)
    
    st.bar_chart(prob_df.set_index('Jenis Kendaraan'))