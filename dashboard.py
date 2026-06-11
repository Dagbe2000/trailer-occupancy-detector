import streamlit as st
from ultralytics import YOLO
import serial
import time
import os
from datetime import datetime

st.set_page_config(page_title="Trailer Occupancy Detector", layout="wide")
st.title("🚛 SkyBitz — Trailer Occupancy Detector")
st.markdown("**Edge ML · YOLOv8-nano · Arduino Nano 33 BLE · Real-time**")

PORT = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200
MODEL_PATH = os.environ.get("MODEL_PATH", "runs/classify/occupancy_detector/weights/best.pt")
SAVE_PATH  = "live_frame.jpg"

# State colors
COLORS = {
    "EMPTY":   "🟢",
    "PARTIAL": "🟡",
    "FULL":    "🔴"
}

# Session state
if "running"    not in st.session_state: st.session_state.running    = False
if "history"    not in st.session_state: st.session_state.history    = []
if "events"     not in st.session_state: st.session_state.events     = 0
if "frames"     not in st.session_state: st.session_state.frames     = 0
if "last_state" not in st.session_state: st.session_state.last_state = None

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Live Camera Feed")
    img_placeholder = st.empty()

with col2:
    st.subheader("📊 Occupancy State")
    state_placeholder  = st.empty()
    conf_placeholder   = st.empty()
    metrics_placeholder = st.empty()

st.subheader("📋 State Change History")
history_placeholder = st.empty()

# Controls
start = st.button("▶️ Start Detection", type="primary")
stop  = st.button("⏹️ Stop")

if start:
    st.session_state.running = True

if stop:
    st.session_state.running = False

if st.session_state.running:
    try:
        model = YOLO(MODEL_PATH)
        ser   = serial.Serial(PORT, BAUD, timeout=10)
        time.sleep(2)

        # Wait for Arduino
        start_t = time.time()
        while time.time() - start_t < 5:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if "READY" in line:
                    break

        while st.session_state.running:
            # Capture frame
            ser.write(b'c')
            all_bytes = bytearray()
            while True:
                chunk = ser.read(1)
                if not chunk:
                    break
                all_bytes += chunk
                if all_bytes.endswith(b'##DONE##'):
                    all_bytes = all_bytes[:-8]
                    break

            jpeg_start = all_bytes.find(b'\xff\xd8')
            jpeg_end   = all_bytes.rfind(b'\xff\xd9')

            if jpeg_start != -1 and jpeg_end != -1:
                jpeg_data = all_bytes[jpeg_start:jpeg_end+2]
                with open(SAVE_PATH, "wb") as f:
                    f.write(jpeg_data)

                # Run inference
                results = model(SAVE_PATH, verbose=False)
                top1    = results[0].probs.top1
                names   = results[0].names
                state   = names[top1].upper()
                conf    = results[0].probs.top1conf.item()

                st.session_state.frames += 1

                # Update camera feed
                img_placeholder.image(SAVE_PATH, caption=f"Frame #{st.session_state.frames}", use_column_width=True)

                # Update state display
                emoji = COLORS.get(state, "⚪")
                state_placeholder.markdown(f"## {emoji} {state}")
                conf_placeholder.progress(conf, text=f"Confidence: {conf:.1%}")

                # State machine
                if state != st.session_state.last_state:
                    st.session_state.events += 1
                    st.session_state.history.append({
                        "time":  datetime.now().strftime("%H:%M:%S"),
                        "from":  st.session_state.last_state or "START",
                        "to":    state,
                        "conf":  f"{conf:.1%}"
                    })
                    st.session_state.last_state = state

                # Metrics
                reduction = 0 if st.session_state.frames == 0 else \
                            (1 - st.session_state.events / st.session_state.frames) * 100
                metrics_placeholder.metric("Frames",       st.session_state.frames)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Frames",  st.session_state.frames)
                m2.metric("State Events",  st.session_state.events)
                m3.metric("MSG Reduction", f"{reduction:.0f}%")

                # History table
                if st.session_state.history:
                    history_placeholder.table(st.session_state.history[-10:])

            time.sleep(0.5)

        ser.close()

    except Exception as e:
        st.error(f"Error: {e}")
