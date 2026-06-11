import serial
import time

import os
PORT = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200
OUTPUT_FILE = "capture.jpg"

ser = serial.Serial(PORT, BAUD, timeout=15)
time.sleep(2)

# Wait for READY
print("Waiting for Arduino...")
while True:
    line = ser.readline().decode(errors='ignore').strip()
    if line:
        print(f"Arduino: {line}")
    if "READY" in line:
        break

# Send capture command
print("Sending capture command...")
ser.write(b'c')
time.sleep(0.1)

# Read raw bytes until ##DONE##
print("Receiving image data...")
all_bytes = bytearray()

while True:
    chunk = ser.read(1)
    if not chunk:
        print("Timeout!")
        break
    all_bytes += chunk
    if all_bytes.endswith(b'##DONE##'):
        all_bytes = all_bytes[:-8]  # Strip marker
        break

print(f"Total bytes received: {len(all_bytes)}")

# Extract JPEG
jpeg_start = all_bytes.find(b'\xff\xd8')
jpeg_end = all_bytes.rfind(b'\xff\xd9')

if jpeg_start != -1 and jpeg_end != -1:
    jpeg_data = all_bytes[jpeg_start:jpeg_end+2]
    with open(OUTPUT_FILE, "wb") as f:
        f.write(jpeg_data)
    print(f"✅ Saved! ({len(jpeg_data)} bytes)")
    os.system(f"xdg-open {OUTPUT_FILE}")
else:
    print(f"❌ JPEG not found. Got {len(all_bytes)} bytes")
    print(f"First 20 bytes: {all_bytes[:20].hex()}")

ser.close()