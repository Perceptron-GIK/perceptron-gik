# This script reads IMU data from the Arduino and writes the data to a CSV file

import time
import imu
from machine import Pin, I2C

bus = I2C(1, scl=Pin(15), sda=Pin(14))
bmi270_bmm150 = imu.IMU(bus)

f = open("imu_raw.csv", "w")
f.write("t_ms,ax,ay,az,gx,gy,gz,mx,my,mz\n")

buffer = []
BUFFER_LIMIT = 256

try:
    while True:
        t = time.ticks_ms()

        # Read IMU data
        ax, ay, az = bmi270_bmm150.accel()
        gx, gy, gz = bmi270_bmm150.gyro()
        mx, my, mz = bmi270_bmm150.magnet()

        # Convert acceleration to m/s^2
        ax *= 9.81
        ay *= 9.81
        az *= 9.81

        # Store IMU data in buffer
        line = "{},{},{},{},{},{},{},{},{},{}\n".format(
            t, ax, ay, az, gx, gy, gz, mx, my, mz
        )
        buffer.append(line)

        # Write IMU data to CSV file
        if len(buffer) >= BUFFER_LIMIT:
            for line in buffer:
                f.write(line)
            buffer = []
            f.flush()

        time.sleep_ms(5)
finally:
    if buffer:
        for line in buffer:
            f.write(line)
        f.flush()
    f.close()