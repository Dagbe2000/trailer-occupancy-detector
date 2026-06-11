import serial
import time
import os
import argparse

PORT = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, choices=["empty", "partial", "full"])
parser.add_argument("--count", type=int, default=50)
args = parser.parse_args()

save_dir = os.path.join("data", "samples", args.label)
os.makedirs(save_dir, exist_ok=True)

existing = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
print(f"Existing images: {existing}")
print(f"Capturing {args.count} images for class: {args.label}")

ser = serial.Serial(PORT, BAUD, timeout=5)
time.sleep(2)

# Wait for READY with timeout
print("Waiting for Arduino (10s timeout)...")
start = time.time()
ready = False
while time.time() - start < 10:
    if ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        print(f"Arduino says: '{line}'")
        if "READY" in line:
            ready = True
            print("Arduino ready!")
            break

# Proceed anyway even without READY
if not ready:
    print("No READY received — proceeding anyway...")

captured = 0
target = args.count

while captured < target:
    idx = existing + captured + 1
    filename = os.path.join(save_dir, f"{args.label}_{idx:04d}.jpg")

    input(f"\n[{captured+1}/{target}] Press ENTER to capture...")

    ser.write(b'c')
    time.sleep(0.1)

    all_bytes = bytearray()
    while True:
        chunk = ser.read(1)
        if not chunk:
            print("Timeout!")
            break
        all_bytes += chunk
        if all_bytes.endswith(b'##DONE##'):
            all_bytes = all_bytes[:-8]
            break

    jpeg_start = all_bytes.find(b'\xff\xd8')
    jpeg_end = all_bytes.rfind(b'\xff\xd9')

    if jpeg_start != -1 and jpeg_end != -1:
        jpeg_data = all_bytes[jpeg_start:jpeg_end+2]
        with open(filename, "wb") as f:
            f.write(jpeg_data)
        print(f"✅ Saved {filename} ({len(jpeg_data)} bytes)")
        captured += 1
    else:
        print(f"❌ Bad image ({len(all_bytes)} bytes), retrying...")

ser.close()
print(f"\n🎉 Done! {captured} images captured for '{args.label}'")
