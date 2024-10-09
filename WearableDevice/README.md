# Adafruit Feather Sense Setup MacOS and Windows

## From Datasheet: https://cdn-learn.adafruit.com/downloads/pdf/adafruit-feather-sense.pdf

1. Arduino Support Setup (p. 37)
- BSP Installation:
  - Download and install the Arduino IDE (https://adafru.it/fvm) (At least v1.8)
  - Start the Arduino IDE
  - Go into Preferences
  - Add https://adafruit.github.io/arduino-board-index/ package_adafruit_index.json as an 'Additional Board Manager URL'.
  - Restart the Arduino IDE
  - Open the Boards Manager option from the Tools -> Board menu and install 'Adafruit nRF52 by Adafruit' 
  - Once the BSP is installed, select from Tools -> Board: Adafruit Bluefruit nRF52840 Feather Sense

2. Arduino Board Testing
- Run a Test Sketch (p.43): Arduino IDE/test_sketch.ino

This will blink the red LED beside the USB port on the Feather, or the red LED labeled
"LED" by the corner of the USB connector on the CLUE.

If Arduino failed to upload sketch to the Feather look at the error codes at p.44. 

The error code "Arduino failed to upload sketch to the Feather" was caused by the bootloader version mismatched on your Feather and installed BSP. To update bootloader, follow this link: https://learn.adafruit.com/bluefruit-nrf52-feather-learning-guide/updating-the-bootloader

Run the code again after bootloader update. 

- Run a Sensor Test (p.45): Arduino IDE/test_sensors.ino

Download all the required libraries for Adafruit: APDS9960, BMP280, LIS3MDL, LSM6D and SHT31

The sensor data are shown under the next section. 

## Sensor Data
Sensor data collected from Arduino IDE/test_sensors.ino

Feather Sense Sensor Demo
---------------------------------------------
- Proximity: 0
- Red: 2 Green: 1 Blue :1 Clear: 3
- Temperature: 21.82 C
- Barometric pressure: 97976.58
- Altitude: 282.58 m
- Magnetic: -1918.00 -610.00 -6895.00 uTesla
- Acceleration: 6.25 -8.42 0.40 m/s^2
- Gyro: -0.18 -0.07 0.00 dps
- Humidity: 51.21 %
- Mic: 626
