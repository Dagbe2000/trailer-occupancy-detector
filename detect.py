from ultralytics import YOLO
import serial
import time
import os

import os
PORT = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200
MODEL_PATH = os.environ.get("MODEL_PATH", "runs/classify/occupancy_detector/weights/best.pt")
SAVE_PATH = "live_frame.jpg"

model = YOLO(MODEL_PATH)
print("✅ Model loaded!")

ser = serial.Serial(PORT, BAUD, timeout=10)
time.sleep(2)

# Wait for Arduino with timeout — proceed anyway
print("Waiting for Arduino (5s timeout)...")
start = time.time()
while time.time() - start < 5:
    if ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"Arduino: {line}")
        if "READY" in line:
            print("✅ Arduino ready!")
            break
print("Proceeding...\n")

last_state = None
events = 0

print("🚀 Live detection started! Press Ctrl+C to stop\n")

while True:
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

        results = model(SAVE_PATH, verbose=False)
        top1  = results[0].probs.top1
        names = results[0].names
        state = names[top1].upper()
        conf  = results[0].probs.top1conf.item()

        if state != last_state:
            events += 1
            print(f"🔔 STATE CHANGE [{events}]: {last_state} → {state} (conf: {conf:.2f})")
            last_state = state
        else:
            print(f"   {state} (conf: {conf:.2f})")

    time.sleep(0.5)
