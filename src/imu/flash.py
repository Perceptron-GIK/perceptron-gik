# This script reads IMU data from the Arduino and writes the data to a CSV file

import time
import imu
from machine import Pin, I2C

bus = I2C(1, scl=Pin(15), sda=Pin(14))
bmi270_bmm150 = imu.IMU(bus)

f = open("imu_raw.csv", "w")
f.write("t_ms,ax,ay,az,gx,gy,gz,mx,my,mz\n")

try:
    while True:
        t = time.ticks_ms() # time since boot

        # Read IMU data
        ax, ay, az = bmi270_bmm150.accel()
        gx, gy, gz = bmi270_bmm150.gyro()
        mx, my, mz = bmi270_bmm150.magnet()

        # Write IMU data to CSV file
        line = "{},{},{},{},{},{},{},{},{},{}\n".format(
            t, ax, ay, az, gx, gy, gz, mx, my, mz
        )
        f.write(line)
        f.flush()

        time.sleep_ms(10)
finally:
    f.close()