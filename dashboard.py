"""
Trailer Occupancy Detector — Streamlit Dashboard
DEMO MODE : Cloud-compatible. Simulated camera, no hardware needed.
LIVE MODE : Local only. Real Arduino + YOLOv8.
"""

import io
import os
import random
import time
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ── Optional live-mode imports ───────────────────────────────────────────────
try:
    import serial as _serial
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

# Auto-detect: demo if hardware libs missing, live if both present
# Override with DEMO_MODE=1 env var to force demo anywhere
DEMO_MODE = (
    os.environ.get("DEMO_MODE", "0") == "1"
    or not _SERIAL_OK
    or not _YOLO_OK
)

PORT       = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
BAUD       = 115200
MODEL_PATH = os.environ.get("MODEL_PATH", "runs/classify/occupancy_detector/weights/best.pt")
SAVE_PATH  = "live_frame.jpg"

STATE_EMOJI = {"EMPTY": "🟢", "PARTIAL": "🟡", "FULL": "🔴"}
STATE_COLOR = {"EMPTY": "#4CAF50", "PARTIAL": "#FF9800", "FULL": "#F44336"}

DEMO_SCENARIO = [
    ("EMPTY",   30, (0.93, 0.99)),
    ("PARTIAL", 20, (0.87, 0.96)),
    ("FULL",    25, (0.91, 0.98)),
    ("PARTIAL", 16, (0.85, 0.94)),
    ("EMPTY",   22, (0.92, 0.99)),
]
_SCENARIO_LEN = sum(c for _, c, _ in DEMO_SCENARIO)


# ── Synthetic frame generators ───────────────────────────────────────────────

def _base() -> np.ndarray:
    img  = Image.new("RGB", (320, 240), (38, 33, 28))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 320, 78], fill=(24, 24, 29))
    for y in range(120, 240):
        t = (y - 120) / 120
        draw.line([(0,y),(320,y)], fill=(int(58+38*t), int(48+28*t), int(38+18*t)))
    draw.rectangle([0, 78, 18, 240],    fill=(48,43,38))
    draw.rectangle([302, 78, 320, 240], fill=(48,43,38))
    draw.line([(0,78),(160,58)],   fill=(68,63,58), width=1)
    draw.line([(320,78),(160,58)], fill=(68,63,58), width=1)
    arr = np.array(img, dtype=np.int16)
    arr += np.random.randint(-8, 9, arr.shape, dtype=np.int16)
    return np.clip(arr, 0, 255).astype(np.uint8)

def _box(draw, x, y, w, h):
    r,g,b = random.randint(145,210), random.randint(125,175), random.randint(95,145)
    draw.rectangle([x,y,x+w,y+h], fill=(r,g,b))
    draw.rectangle([x,y-h//4,x+w,y], fill=(min(r+45,255),min(g+40,255),min(b+35,255)))
    draw.rectangle([x,y-h//4,x+w,y+h], outline=(18,18,18), width=1)

def gen_frame(state: str) -> np.ndarray:
    arr  = _base()
    if state == "EMPTY":
        return arr
    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    if state == "PARTIAL":
        bw,bh = random.randint(54,74), random.randint(40,57)
        for ry in range(128, 218, bh+3):
            for rx in range(20, random.randint(140,175), bw+3):
                _box(draw, rx, ry, bw, bh)
    else:  # FULL
        bw,bh = random.randint(50,68), random.randint(38,54)
        for ry in range(96, 222, bh+3):
            for rx in range(20, 296, bw+3):
                _box(draw, rx, ry, bw, bh)
    return np.array(img)

def demo_state(frame_idx: int):
    pos = frame_idx % _SCENARIO_LEN
    for state, count, (lo, hi) in DEMO_SCENARIO:
        if pos < count:
            return state, round(random.uniform(lo, hi), 3)
        pos -= count
    return "EMPTY", 0.95


# ── Page ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Trailer Occupancy Detector", layout="wide", page_icon="🚛")
st.title("🚛 SkyBitz — Trailer Occupancy Detector")
st.markdown("**Edge ML · YOLOv8-nano · Arduino Nano 33 BLE · Real-time**")
st.caption("🎬 Demo Mode — simulated camera" if DEMO_MODE else "🔌 Live Mode — Arduino connected")

with st.sidebar:
    st.header("⚙️ Settings")
    if DEMO_MODE:
        st.info("🎬 **Demo Mode**\nSimulated camera & inference.\nNo hardware required.")
        speed = st.slider("Speed (fps)", 0.5, 3.0, 1.0, 0.5)
    else:
        st.success("🔌 **Live Mode**")
        speed = 2.0
    st.divider()
    st.markdown("**Model** YOLOv8-nano cls")
    st.markdown("**Accuracy** 94.4 %")
    st.markdown("**Inference** 6.9 ms avg")
    st.markdown("**MSG reduction** ~95 %")
    st.divider()
    st.caption("Wilfrid Hounkponou (Zoyo)\nhounkponou@cua.edu")

# Session state
for k, v in [("running",False),("history",[]),("events",0),("frames",0),("last_state",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# Controls
b1, b2 = st.columns(2)
if b1.button("▶️ Start Detection", type="primary", use_container_width=True):
    st.session_state.update(running=True, history=[], events=0, frames=0, last_state=None)
if b2.button("⏹️ Stop", use_container_width=True):
    st.session_state.running = False
    if "live_ser" in st.session_state:
        try: st.session_state.live_ser.close()
        except Exception: pass
        del st.session_state.live_ser


# ── Demo mode ────────────────────────────────────────────────────────────────
if st.session_state.running and DEMO_MODE:

    # Advance one frame and cache it in session state
    state, conf = demo_state(st.session_state.frames)
    img         = gen_frame(state)
    st.session_state.frames    += 1
    st.session_state.cur_img    = img
    st.session_state.cur_state  = state
    st.session_state.cur_conf   = conf

    if state != st.session_state.last_state:
        st.session_state.events += 1
        st.session_state.history.append({
            "Time":       datetime.now().strftime("%H:%M:%S"),
            "Transition": f"{st.session_state.last_state or 'START'} → {state}",
            "Confidence": f"{conf:.1%}",
        })
        st.session_state.last_state = state

# Show cached frame (visible on current AND next rerun)
if "cur_img" in st.session_state and (st.session_state.running or st.session_state.frames > 0):
    state = st.session_state.cur_state
    conf  = st.session_state.cur_conf

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Camera Feed")
        st.image(st.session_state.cur_img,
                 caption=f"Frame #{st.session_state.frames}",
                 use_container_width=True)

    with col2:
        st.subheader("📊 Occupancy State")
        color = STATE_COLOR.get(state, "#9E9E9E")
        emoji = STATE_EMOJI.get(state, "⚪")
        st.markdown(
            f"<div style='text-align:center;padding:20px;background:{color}22;"
            f"border:2px solid {color};border-radius:10px;margin-bottom:12px'>"
            f"<div style='font-size:3em'>{emoji}</div>"
            f"<div style='font-size:2em;font-weight:bold;color:{color}'>{state}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(conf, text=f"Confidence: {conf:.1%}")
        st.divider()
        reduction = 0 if st.session_state.frames == 0 else \
                    (1 - st.session_state.events / st.session_state.frames) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Frames",  st.session_state.frames)
        m2.metric("Events",  st.session_state.events)
        m3.metric("MSG ↓",   f"{reduction:.0f}%")

    if st.session_state.history:
        st.subheader("📋 State Change History")
        st.table(st.session_state.history[-10:])

# Schedule next frame AFTER rendering (sleep here controls visible frame rate)
if st.session_state.running and DEMO_MODE:
    time.sleep(1.0 / speed)
    st.rerun()


# ── Live mode (local only) ───────────────────────────────────────────────────
if st.session_state.running and not DEMO_MODE:
    try:
        # Open serial once; persist across reruns
        if "live_ser" not in st.session_state:
            st.session_state.live_ser = _serial.Serial(PORT, BAUD, timeout=10)
            time.sleep(2)

        # Load model once; persist across reruns
        if "live_model" not in st.session_state:
            st.session_state.live_model = _YOLO(MODEL_PATH)

        ser = st.session_state.live_ser
        ser.write(b"c")
        buf = bytearray()
        while True:
            chunk = ser.read(1)
            if not chunk: break
            buf += chunk
            if buf.endswith(b"##DONE##"):
                buf = buf[:-8]; break

        js, je = buf.find(b"\xff\xd8"), buf.rfind(b"\xff\xd9")
        if js != -1 and je != -1:
            live_img = np.array(Image.open(io.BytesIO(buf[js:je+2])).convert("RGB"))
            res   = st.session_state.live_model(live_img, verbose=False)
            state = res[0].names[res[0].probs.top1].upper()
            conf  = res[0].probs.top1conf.item()

            st.session_state.cur_img   = live_img
            st.session_state.cur_state = state
            st.session_state.cur_conf  = conf
            st.session_state.frames   += 1

            if state != st.session_state.last_state:
                st.session_state.events += 1
                st.session_state.history.append({
                    "Time":       datetime.now().strftime("%H:%M:%S"),
                    "Transition": f"{st.session_state.last_state or 'START'} → {state}",
                    "Confidence": f"{conf:.1%}",
                })
                st.session_state.last_state = state

        time.sleep(0.5)
        st.rerun()

    except Exception as e:
        st.error(f"Live mode error: {e}")
        st.session_state.running = False
        if "live_ser" in st.session_state:
            try: st.session_state.live_ser.close()
            except Exception: pass
            del st.session_state.live_ser
