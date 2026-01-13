import time
import imu
from machine import Pin, I2C

bus = I2C(1, scl=Pin(15), sda=Pin(14))
m = imu.IMU(bus)

while True:
    print("Accel:", m.accel())
    print("Gyro:", m.gyro())
    print("Mag:", m.magnet())
    print()
    time.sleep_ms(10)