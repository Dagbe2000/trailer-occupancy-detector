/*
  Trailer Occupancy Detector — Arduino Firmware
  -----------------------------------------------
  Hardware : Arduino Nano 33 BLE Rev2 + Arducam Mini OV2640
  Author   : Wilfrid Hounkponou (Zoyo) <hounkponou@cua.edu>
  Project  : Edge ML cargo sensing for SkyBitz/AMETEK IoT fleet management

  Wiring (OV2640 → Nano 33 BLE):
    CS   → D10   MOSI → D11   MISO → D12   SCK → D13
    SDA  → A4    SCL  → A5    VCC  → 3.3V  GND → GND

  Protocol:
    Host sends 'c' over Serial (115200 baud)
    Arduino captures JPEG, streams raw bytes, ends with "##DONE##"
*/

#include <ArduCAM.h>
#include <SPI.h>
#include <Wire.h>
#include "memorysaver.h"

const int CS = 10;
ArduCAM myCAM(OV2640, CS);

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Wire.begin();
  SPI.begin();
  pinMode(CS, OUTPUT);
  digitalWrite(CS, HIGH);

  // Reset camera
  myCAM.write_reg(0x07, 0x80);
  delay(100);
  myCAM.write_reg(0x07, 0x00);
  delay(100);

  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_320x240);
  delay(500);

  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'c') {
      captureAndSend();
    }
  }
}

void captureAndSend() {
  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();

  // Wait for capture to complete (with timeout)
  unsigned long t = millis();
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() - t > 3000) {
      Serial.println("##DONE##");
      return;
    }
  }

  uint32_t length = myCAM.read_fifo_length();
  if (length == 0 || length > 0x5FFFF) {
    Serial.println("##DONE##");
    return;
  }

  myCAM.CS_LOW();
  myCAM.set_fifo_burst();

  for (uint32_t i = 0; i < length; i++) {
    Serial.write(SPI.transfer(0x00));
  }

  myCAM.CS_HIGH();
  Serial.flush();
  Serial.println("##DONE##");
}
