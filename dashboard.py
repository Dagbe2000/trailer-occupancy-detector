"""
Trailer Occupancy Detector — Streamlit Dashboard

DEMO MODE  : Simulated camera + inference. No hardware needed.
             Runs on Streamlit Community Cloud.
LIVE MODE  : Real Arduino + YOLOv8 model. Requires hardware.

Mode is auto-detected (falls back to demo if serial/ultralytics unavailable)
or forced via DEMO_MODE=1 environment variable.
"""

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

DEMO_MODE = (
    os.environ.get("DEMO_MODE", "1") == "1"
    or not _SERIAL_OK
    or not _YOLO_OK
)

# ── Config ───────────────────────────────────────────────────────────────────
PORT       = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
BAUD       = 115200
MODEL_PATH = os.environ.get("MODEL_PATH", "runs/classify/occupancy_detector/weights/best.pt")
SAVE_PATH  = "live_frame.jpg"

STATE_EMOJI = {"EMPTY": "🟢", "PARTIAL": "🟡", "FULL": "🔴"}
STATE_COLOR = {"EMPTY": "#4CAF50", "PARTIAL": "#FF9800", "FULL": "#F44336"}

# Demo scenario: list of (state, frame_count, (conf_min, conf_max))
# Cycles continuously — models a realistic load/unload sequence
DEMO_SCENARIO = [
    ("EMPTY",   30, (0.93, 0.99)),   # trailer arrives empty
    ("PARTIAL", 20, (0.87, 0.96)),   # loading begins
    ("FULL",    25, (0.91, 0.98)),   # fully loaded
    ("PARTIAL", 16, (0.85, 0.94)),   # unloading begins
    ("EMPTY",   22, (0.92, 0.99)),   # trailer empty again
]
_SCENARIO_LEN = sum(c for _, c, _ in DEMO_SCENARIO)


# ── Synthetic frame generators (PIL, no OpenCV) ──────────────────────────────

def _base_interior() -> Image.Image:
    """Draw a bare trailer interior: floor, ceiling, side walls, vanishing lines."""
    img = Image.new("RGB", (320, 240), (38, 33, 28))
    draw = ImageDraw.Draw(img)
    # Ceiling
    draw.rectangle([0, 0, 320, 78], fill=(24, 24, 29))
    # Floor gradient rows
    for y in range(120, 240):
        t = (y - 120) / 120
        c = (int(58 + 38 * t), int(48 + 28 * t), int(38 + 18 * t))
        draw.line([(0, y), (320, y)], fill=c)
    # Walls
    draw.rectangle([0, 78, 18, 240], fill=(48, 43, 38))
    draw.rectangle([302, 78, 320, 240], fill=(48, 43, 38))
    # Perspective vanish lines
    draw.line([(0, 78), (160, 58)],  fill=(68, 63, 58), width=1)
    draw.line([(320, 78), (160, 58)], fill=(68, 63, 58), width=1)
    # Subtle noise
    arr = np.array(img, dtype=np.int16)
    arr += np.random.randint(-8, 9, arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _draw_box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    r = random.randint(145, 210)
    g = random.randint(125, 175)
    b = random.randint(95, 145)
    draw.rectangle([x, y, x + w, y + h], fill=(r, g, b))
    draw.rectangle([x, y - h // 4, x + w, y],
                   fill=(min(r + 45, 255), min(g + 40, 255), min(b + 35, 255)))
    draw.rectangle([x, y - h // 4, x + w, y + h], outline=(18, 18, 18), width=1)


def _gen_empty() -> Image.Image:
    return _base_interior()


def _gen_partial() -> Image.Image:
    img  = _base_interior()
    draw = ImageDraw.Draw(img)
    bw, bh = random.randint(54, 74), random.randint(40, 57)
    side_end = random.randint(140, 175)
    for ry in range(128, 218, bh + random.randint(2, 5)):
        for rx in range(20, side_end, bw + random.randint(2, 4)):
            _draw_box(draw, rx, ry, bw, bh)
    return img


def _gen_full() -> Image.Image:
    img  = _base_interior()
    draw = ImageDraw.Draw(img)
    bw, bh = random.randint(50, 68), random.randint(38, 54)
    for ry in range(96, 222, bh + random.randint(2, 4)):
        for rx in range(20, 296, bw + random.randint(2, 4)):
            _draw_box(draw, rx, ry, bw, bh)
    return img


_GEN = {"EMPTY": _gen_empty, "PARTIAL": _gen_partial, "FULL": _gen_full}


def _demo_frame(frame_idx: int) -> tuple[Image.Image, str, float]:
    """Return (PIL image, state, confidence) for a given frame index."""
    pos = frame_idx % _SCENARIO_LEN
    for state, count, (lo, hi) in DEMO_SCENARIO:
        if pos < count:
            return _GEN[state](), state, round(random.uniform(lo, hi), 3)
        pos -= count
    return _gen_empty(), "EMPTY", 0.95


# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Trailer Occupancy Detector", layout="wide", page_icon="🚛")
st.title("🚛 SkyBitz — Trailer Occupancy Detector")
st.markdown("**Edge ML · YOLOv8-nano · Arduino Nano 33 BLE · Real-time**")
st.caption(f"{'🎬 Demo Mode — simulated camera' if DEMO_MODE else '🔌 Live Mode — Arduino connected'}")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Mode")
    if DEMO_MODE:
        st.info("🎬 **Demo Mode**\nSimulated camera & inference.\nNo hardware required.")
        fps = st.slider("Simulation speed (fps)", 0.5, 5.0, 2.0, 0.5)
    else:
        st.success("🔌 **Live Mode**\nArduino connected.")
        fps = 2.0

    st.divider()
    st.markdown("**Model** YOLOv8-nano cls")
    st.markdown("**Classes** EMPTY · PARTIAL · FULL")
    st.markdown("**Accuracy** 94.4 %")
    st.markdown("**Inference** 6.9 ms avg")
    st.markdown("**MSG reduction** ~95 %")
    st.divider()
    st.caption("Wilfrid Hounkponou (Zoyo)\nhounkponou@cua.edu")

# ── Session state init ───────────────────────────────────────────────────────
for k, v in [("running", False), ("history", []), ("events", 0),
             ("frames", 0), ("last_state", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Layout placeholders ──────────────────────────────────────────────────────
col_cam, col_state = st.columns([1, 1])
with col_cam:
    st.subheader("📷 Camera Feed")
    img_ph = st.empty()
with col_state:
    st.subheader("📊 Occupancy State")
    state_ph = st.empty()
    conf_ph  = st.empty()
    st.markdown("---")
    st.markdown("**Performance metrics**")
    mc1, mc2, mc3 = st.columns(3)
    frames_ph    = mc1.empty()
    events_ph    = mc2.empty()
    reduction_ph = mc3.empty()

st.subheader("📋 State Change History")
history_ph = st.empty()

# ── Controls ─────────────────────────────────────────────────────────────────
btn1, btn2 = st.columns(2)
if btn1.button("▶️ Start Detection", type="primary", use_container_width=True):
    st.session_state.running    = True
    st.session_state.history    = []
    st.session_state.events     = 0
    st.session_state.frames     = 0
    st.session_state.last_state = None

if btn2.button("⏹️ Stop", use_container_width=True):
    st.session_state.running = False


# ── Shared UI update ─────────────────────────────────────────────────────────

def _update(image, state: str, conf: float) -> None:
    img_ph.image(image, caption=f"Frame #{st.session_state.frames}",
                 use_container_width=True)

    color = STATE_COLOR.get(state, "#9E9E9E")
    emoji = STATE_EMOJI.get(state, "⚪")
    state_ph.markdown(
        f"<div style='text-align:center;padding:18px;background:{color}22;"
        f"border:2px solid {color};border-radius:10px'>"
        f"<span style='font-size:3em'>{emoji}</span><br>"
        f"<span style='font-size:1.8em;font-weight:bold;color:{color}'>{state}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    conf_ph.progress(conf, text=f"Confidence: {conf:.1%}")

    if state != st.session_state.last_state:
        st.session_state.events += 1
        st.session_state.history.append({
            "Time":        datetime.now().strftime("%H:%M:%S"),
            "Transition":  f"{st.session_state.last_state or 'START'} → {state}",
            "Confidence":  f"{conf:.1%}",
        })
        st.session_state.last_state = state

    reduction = (
        0 if st.session_state.frames == 0
        else (1 - st.session_state.events / st.session_state.frames) * 100
    )
    frames_ph.metric("Frames",     st.session_state.frames)
    events_ph.metric("Events",     st.session_state.events)
    reduction_ph.metric("MSG ↓",  f"{reduction:.0f}%")

    if st.session_state.history:
        history_ph.table(st.session_state.history[-10:])


# ── Main detection loop ───────────────────────────────────────────────────────

if st.session_state.running:

    if DEMO_MODE:
        status = st.empty()
        status.info("🎬 Simulating — close tab or click Stop to end.")
        try:
            while st.session_state.running:
                img, state, conf = _demo_frame(st.session_state.frames)
                st.session_state.frames += 1
                _update(img, state, conf)
                time.sleep(1.0 / fps)
        except Exception as exc:
            st.error(f"Demo error: {exc}")
        finally:
            st.session_state.running = False
            status.empty()

    else:
        try:
            model = _YOLO(MODEL_PATH)
            ser   = _serial.Serial(PORT, BAUD, timeout=10)
            time.sleep(2)

            t0 = time.time()
            while time.time() - t0 < 5:
                if ser.in_waiting:
                    if "READY" in ser.readline().decode(errors="ignore"):
                        break

            while st.session_state.running:
                ser.write(b"c")
                buf = bytearray()
                while True:
                    chunk = ser.read(1)
                    if not chunk:
                        break
                    buf += chunk
                    if buf.endswith(b"##DONE##"):
                        buf = buf[:-8]
                        break

                js = buf.find(b"\xff\xd8")
                je = buf.rfind(b"\xff\xd9")
                if js != -1 and je != -1:
                    with open(SAVE_PATH, "wb") as f:
                        f.write(buf[js:je + 2])
                    res   = model(SAVE_PATH, verbose=False)
                    state = res[0].names[res[0].probs.top1].upper()
                    conf  = res[0].probs.top1conf.item()
                    st.session_state.frames += 1
                    _update(SAVE_PATH, state, conf)

                time.sleep(0.5)

            ser.close()

        except Exception as exc:
            st.error(f"Connection error: {exc}")
            st.session_state.running = False
