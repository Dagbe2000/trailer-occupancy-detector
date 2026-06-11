import streamlit as st
import os
from PIL import Image
import serial
import time

st.set_page_config(page_title="Occupancy Detector", layout="wide")
st.title("🚛 Trailer Occupancy Detector — Live Viewer")

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")
PORT = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200

# ── Dataset Stats ──────────────────────────────────
st.header("📊 Dataset Progress")
cols = st.columns(3)
for i, label in enumerate(["empty", "partial", "full"]):
    folder = f"{SAMPLES_DIR}/{label}"
    count = len(os.listdir(folder)) if os.path.exists(folder) else 0
    cols[i].metric(label.upper(), f"{count} images", f"{count}/50")

# ── Live Capture ───────────────────────────────────
st.header("📸 Live Capture")
label = st.selectbox("Select Class", ["empty", "partial", "full"])

if st.button("📷 Capture Image"):
    try:
        ser = serial.Serial(PORT, BAUD, timeout=10)
        time.sleep(1)
        ser.write(b'c')
        time.sleep(0.1)

        all_bytes = bytearray()
        while True:
            chunk = ser.read(1)
            if not chunk:
                st.error("Timeout!")
                break
            all_bytes += chunk
            if all_bytes.endswith(b'##DONE##'):
                all_bytes = all_bytes[:-8]
                break
        ser.close()

        jpeg_start = all_bytes.find(b'\xff\xd8')
        jpeg_end   = all_bytes.rfind(b'\xff\xd9')

        if jpeg_start != -1 and jpeg_end != -1:
            jpeg_data = all_bytes[jpeg_start:jpeg_end+2]
            save_dir  = f"{SAMPLES_DIR}/{label}"
            os.makedirs(save_dir, exist_ok=True)
            count     = len(os.listdir(save_dir))
            filename  = f"{save_dir}/{label}_{count+1:04d}.jpg"
            with open(filename, "wb") as f:
                f.write(jpeg_data)
            st.success(f"✅ Saved {filename}")
            st.image(filename, caption=f"Latest: {label}", width=400)
        else:
            st.error("❌ Bad image data")
    except Exception as e:
        st.error(f"Error: {e}")

# ── Gallery ────────────────────────────────────────
st.header("🖼️ Image Gallery")
tab1, tab2, tab3 = st.tabs(["EMPTY", "PARTIAL", "FULL"])

for tab, label in zip([tab1, tab2, tab3], ["empty", "partial", "full"]):
    with tab:
        folder = f"{SAMPLES_DIR}/{label}"
        if os.path.exists(folder):
            images = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
            if images:
                # Show last 9 images in 3x3 grid
                recent = images[-9:]
                cols = st.columns(3)
                for i, img_file in enumerate(recent):
                    with cols[i % 3]:
                        img_path = f"{folder}/{img_file}"
                        st.image(img_path, caption=img_file, use_column_width=True)
            else:
                st.info(f"No images yet for {label}")
        else:
            st.info(f"Folder not found for {label}")
