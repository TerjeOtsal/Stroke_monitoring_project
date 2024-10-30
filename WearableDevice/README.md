# Adafruit Feather Sense Setup (MacOS and Windows)

## Datasheet Reference:
[Adafruit Feather Sense Datasheet](https://cdn-learn.adafruit.com/downloads/pdf/adafruit-feather-sense.pdf)

## 1. Arduino Support Setup (p. 37)

### BSP Installation:

1. Download and install the Arduino IDE (v1.8 or later) from [here](https://adafru.it/fvm).
2. Start the Arduino IDE.
3. Go to **Preferences**.
4. Add the following URL as an 'Additional Board Manager URL':  
   `https://adafruit.github.io/arduino-board-index/package_adafruit_index.json`
5. Restart the Arduino IDE.
6. Open **Boards Manager** from the Tools -> Board menu.
7. Install 'Adafruit nRF52 by Adafruit'.
8. Once the BSP is installed, select **Tools -> Board: Adafruit Bluefruit nRF52840 Feather Sense**.

## 2. Arduino Board Testing

**Note:** `.ino` files need to be run inside the Arduino IDE. Copy and paste the code as needed.

### Run a Test Sketch (p.43):  
`Arduino IDE/test_sketch.ino`

This sketch will blink the red LED beside the USB port on the Feather, or the red LED labeled "LED" by the corner of the USB connector on the CLUE board.

- If Arduino fails to upload the sketch to the Feather, check the error codes on p.44 of the datasheet.  
- The error **"Arduino failed to upload sketch to the Feather"** is often caused by a mismatch between the bootloader version on your Feather and the installed BSP.

To update the bootloader, follow this guide: [Updating the Bootloader](https://learn.adafruit.com/bluefruit-nrf52-feather-learning-guide/updating-the-bootloader).

Once the bootloader is updated, run the code again.

### Run a Sensor Test (p.45):  
`Arduino IDE/test_sensors.ino`

Download the required libraries for Adafruit:  
`APDS9960, BMP280, LIS3MDL, LSM6D, SHT31`

The sensor data will be displayed in the next section.

## 3. Sensor Data

Sensor data collected from `Arduino IDE/test_sensors.ino`:

**Feather Sense Sensor Demo:**

- Proximity: 0
- Red: 2, Green: 1, Blue: 1, Clear: 3
- Temperature: 21.82°C
- Barometric pressure: 97976.58 Pa
- Altitude: 282.58 m
- Magnetic field: -1918.00, -610.00, -6895.00 µTesla
- Acceleration: 6.25, -8.42, 0.40 m/s²
- Gyro: -0.18, -0.07, 0.00 dps
- Humidity: 51.21%
- Microphone: 626

## inertial navigation

## Converting gyro, accelerometer and magnetometer to readable data
https://forums.raspberrypi.com/viewtopic.php?t=127930

- using euler angles means you'll have issues with the gimbal lock and singularities (https://en.wikipedia.org/wiki/Gimbal_lock)
- the magnetometer needs some low pass filtering as it's pretty noisy (+- 1-3 degrees after conversion to angles)

- Vurder complementary filter for relative position: 
  - Relative orientation is the recovery of the position and orientation of one object relative to another. Depending on the direction there are three types of angular rate measurements:

   Yaw: the horizontal rotation on a flat surface when seen the object from above.

   Pitch: vertical rotation as seen the object from front.

   Roll: horizontal rotation when seen the object from front.

   Referansepunkt mobil, gps

   Bruk Quaternion for fri bevegelse

   INS (inertial navigation system) og startknapp for referanse

   ## Environment

   python3 -m venv rehabenv
   source rehabenv/bin/activate

/Users/sunnivajosefsen/Library/CloudStorage/OneDrive-OsloMet/Master/ACIT4035

## Password keyring
If you would like to delete a password stored in your system keyring, you can clear a stored password using the --delete-from-keyring command-line option:

$ icloud --username=jappleseed@apple.com --delete-from-keyring

https://github.com/john2zy/IMU-Position-Tracking

https://github.com/xioTechnologies/Gait-Tracking/tree/main