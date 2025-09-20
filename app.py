import streamlit as st
from pathlib import Path
import tempfile
import cv2
from ultralytics import YOLO
import os
import time
import base64

# ====== PAGE CONFIG ======
st.set_page_config(page_title="AquaFun", layout="wide")

# ====== LOAD CSS ======
css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ====== HEADER ======
st.markdown("""
<div class="header">
    <div class="logo">AquaFun</div>
    <div class="nav">
        <a href="#home">Home</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ====== HERO SECTION ======
st.markdown("""
<div class="hero" id="home">
    <h1>Welcome to AquaFun</h1>
    <p>Discover the fascinating world of fish — from habitats, behaviors, and ecological traits — right from your aquarium videos.</p>
</div>
""", unsafe_allow_html=True)

# ====== UPLOAD SECTION ======
st.markdown("<h2 class='upload-title'>🎥 Share your aquarium video with us!</h2>", unsafe_allow_html=True)
st.markdown("<p class='upload-subtitle'>Upload your aquarium video and let us process it for fish detection!</p>", unsafe_allow_html=True)

uploaded_video = st.file_uploader("", type=["mp4", "mov", "avi", "mkv", "webm", "wmv", "mpeg"])

# ====== LOAD YOLO MODEL ======
model = YOLO("best.pt")  # pastikan file ada di folder proyek

if uploaded_video:
    # Simpan file upload sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Siapkan output video
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("❌ Tidak bisa membuka video. Pastikan formatnya didukung.")
        st.stop()

    # Gunakan codec H.264 agar kompatibel di browser
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_path = os.path.join(tempfile.gettempdir(), "detected.mp4")
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_text = st.empty()
    progress_bar = st.progress(0)

    frame_id = 0
    annotated_frame = None  # simpan hasil terakhir

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hanya deteksi setiap 5 frame
        if frame_id % 5 == 0:
            results = model(frame, imgsz=320, conf=0.5)
            annotated_frame = results[0].plot()

        # Tulis frame (pakai hasil terbaru)
        if annotated_frame is not None:
            out.write(annotated_frame)
        else:
            out.write(frame)

        frame_id += 1
        progress_bar.progress(frame_id / total_frames)
        progress_text.text(f"Processing frame {frame_id}/{total_frames}...")

    # Tutup video
    cap.release()
    out.release()

    # Pastikan file selesai ditulis
    time.sleep(0.5)

    progress_text.text("✅ Detection complete!")

    # Tampilkan video
    st.video(out_path)

    # Tombol download video
    with open(out_path, "rb") as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="detected.mp4">📥 Download Hasil Deteksi</a>'
    st.markdown(href, unsafe_allow_html=True)
