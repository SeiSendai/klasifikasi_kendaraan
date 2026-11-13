import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 1. Definisikan Arsitektur Model yang sama persis seperti saat training
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

# 2. Fungsi untuk Memuat Model dan Kelas
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

# 3. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Vehicle Type Classification", page_icon="🚗", layout="centered")

st.title("🚗 Klasifikasi Jenis Kendaraan (Vehicle Type Classifier)")
st.write("Aplikasi web berbasis PyTorch dan Streamlit untuk mendeteksi kategori kendaraan dari gambar.")

model, class_names = load_model()

if model is None:
    st.error("⚠️ File model `model_custom_cnn.pth` tidak ditemukan! Pastikan Anda sudah menjalankan training dan menyimpan model di direktori yang sama.")
else:
    # 4. Upload Gambar oleh Pengguna
    uploaded_file = st.file_uploader("Pilih gambar kendaraan (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Tampilkan gambar di UI
        st.image(image, caption="Gambar Kendaraan yang Diunggah", use_container_width=True)
        st.write("")
        
        if st.button("🔍 Prediksi Jenis Kendaraan"):
            with st.spinner("Sedang memproses gambar..."):
                # Preprocessing gambar sesuai dengan tahap test_transform
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(image).unsqueeze(0) # Tambahkan dimensi batch
                
                # Prediksi dengan model
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    confidence, predicted_idx = torch.max(probabilities, 0)
                
                predicted_class = class_names[predicted_idx.item()]
                confidence_score = confidence.item() * 100
                
            # 5. Tampilkan Hasil Prediksi
            st.success(f"**Hasil Prediksi:** {predicted_class}")
            st.info(f"**Tingkat Kepercayaan (Confidence):** {confidence_score:.2f}%")
            
            # Tampilkan probabilitas semua kelas
            st.markdown("### Detail Probabilitas per Kelas:")
            for idx, prob in enumerate(probabilities):
                st.write(f"- **{class_names[idx]}**: `{prob.item() * 100:.2f}%`")