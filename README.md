# Trailer Occupancy Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-orange?logo=pytorch)
![Arduino](https://img.shields.io/badge/Arduino-Nano%2033%20BLE-teal?logo=arduino)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

Real-time edge ML system that classifies trailer/container occupancy as **EMPTY**, **PARTIAL**, or **FULL** using a live camera feed and YOLOv8-nano, running inference on a host PC with images captured from an Arduino-connected Arducam OV2640.

---

## Results

| Metric | Value |
|---|---|
| Top-1 Classification Accuracy | **94.4%** |
| Live Inference Confidence | **95.5%** |
| Inference Speed | **6.9 ms / frame** |
| MQTT Message Reduction | **~95%** via state machine |
| Training Images | **90 custom images** (30 per class) |
| Model | YOLOv8-nano classification |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EDGE DEVICE                              │
│                                                                 │
│   ┌──────────────┐    SPI/I2C    ┌───────────────────────┐     │
│   │  Arducam     │──────────────▶│  Arduino Nano 33 BLE  │     │
│   │  OV2640      │               │                       │     │
│   │  320×240 px  │               │  1. Wait for 'c' cmd  │     │
│   └──────────────┘               │  2. Trigger capture   │     │
│                                  │  3. Stream JPEG bytes  │     │
│                                  │  4. Send ##DONE##      │     │
│                                  └──────────┬────────────┘     │
└─────────────────────────────────────────────│────────────────── ┘
                                              │ USB Serial (115200)
┌─────────────────────────────────────────────│────────────────── ┐
│                        HOST PC              │                    │
│                                             ▼                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    detect.py / dashboard.py             │   │
│   │                                                         │   │
│   │  ① Receive JPEG bytes → save live_frame.jpg            │   │
│   │  ② YOLOv8-nano inference  (6.9ms avg)                  │   │
│   │  ③ State machine: publish only on EMPTY/PARTIAL/FULL   │   │
│   │     change  →  ~95% message reduction                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Streamlit Dashboard                        │   │
│   │   Live feed · State badge · Confidence · History table  │   │
│   └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────── ──┘
```

---

## Hardware

| Component | Details |
|---|---|
| Microcontroller | Arduino Nano 33 BLE Rev2 |
| Camera | Arducam Mini Module OV2640 (2MP, SPI/I2C) |
| Interface | USB Serial @ 115200 baud |
| Resolution | 320 × 240 JPEG |
| Host OS | Ubuntu 22.04+ / Windows 10+ |

---

## Wiring

| Arducam Pin | Arduino Nano 33 BLE Pin |
|---|---|
| CS | D10 |
| MOSI | D11 |
| MISO | D12 |
| SCK | D13 |
| SDA | A4 |
| SCL | A5 |
| VCC | 3.3V |
| GND | GND |

---

## Project Structure

```
trailer-occupancy-detector/
├── arduino/
│   └── trailer_occupancy_detector/
│       └── trailer_occupancy_detector.ino   # Firmware: capture on 'c' command
├── data/
│   └── samples/                             # Training images (gitignored)
│       ├── empty/
│       ├── partial/
│       └── full/
├── runs/                                    # Model weights (gitignored)
│   └── classify/occupancy_detector/weights/best.pt
├── dashboard.py                             # Streamlit live dashboard
├── detect.py                                # CLI live inference loop
├── train.py                                 # YOLOv8 training (Roboflow dataset)
├── capture_dataset.py                       # Dataset collection from Arduino
├── receive_image.py                         # Single test capture
├── viewer.py                                # Dataset progress viewer
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/Zoyo-Dev/trailer-occupancy-detector.git
cd trailer-occupancy-detector
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your ROBOFLOW_API_KEY
```

### 3. Flash Arduino firmware
- Open `arduino/trailer_occupancy_detector/trailer_occupancy_detector.ino` in Arduino IDE
- Install **ArduCAM** library via Library Manager
- Select board: **Arduino Nano 33 BLE**, port: `/dev/ttyACM0`
- Upload

### 4. Train the model
```bash
python train.py
# Downloads dataset from Roboflow, trains YOLOv8-nano, saves best.pt
```

### 5. Run live detection (CLI)
```bash
python detect.py
# Connects to Arduino, runs inference, prints state changes
```

### 6. Run Streamlit dashboard
```bash
streamlit run dashboard.py
# Open http://localhost:8501 — click ▶️ Start Detection
```

### 7. Collect your own dataset
```bash
# Capture 30 images per class
python capture_dataset.py --label empty   --count 30
python capture_dataset.py --label partial --count 30
python capture_dataset.py --label full    --count 30
```

---

## How It Works

### 1. Image Capture
The host sends a single byte `'c'` over USB Serial. The Arduino triggers the OV2640, reads the JPEG from the camera's FIFO buffer byte-by-byte over SPI, streams it back, and appends `##DONE##` as a framing marker. The host strips the marker and saves the JPEG.

### 2. Classification
The saved JPEG is passed to a fine-tuned **YOLOv8-nano classification model**. The model outputs probabilities for `[EMPTY, PARTIAL, FULL]`; the top-1 class with confidence is used.

### 3. State Machine (Message Reduction)
A simple state machine compares each frame's classification to the previous committed state. Only when the state changes is an event logged (and would be published to a broker). This produces **~95% message reduction** — a critical property for bandwidth-constrained IoT deployments where trailers may remain in the same state for hours.

### 4. Dashboard
The Streamlit dashboard visualizes the live camera feed, current occupancy state (color-coded 🟢 / 🟡 / 🔴), confidence progress bar, frame/event counters, and a scrollable state-change history table — all updating in real time.

---

## Training Details

- **Base model**: `yolov8n-cls.pt` (YOLOv8-nano, classification head)
- **Dataset**: 90 custom images (30 per class) captured with OV2640 in real trailer/container conditions, labeled and hosted on Roboflow
- **Augmentation**: random flip, HSV jitter, mosaic (via Ultralytics defaults)
- **Epochs**: 50, batch 16, imgsz 320, patience 10

---

## Use Case

This project mirrors the cargo sensing pipeline used in fleet IoT platforms like **SkyBitz / AMETEK**:

- Camera-equipped trailer doors → occupancy detection at load/unload
- Edge inference → no video streaming to cloud, only state events
- State-centric events → minimal cellular data usage over LTE/NB-IoT
- Real-time dashboard → dispatcher visibility into fleet load status

---

## Author

**Wilfrid Hounkponou **
- Email: wilfrid19921992@gmail.com
- GitHub: [@Dagbe2000](https://github.com/Dagbe2000/

---

## License

MIT
