#----------------------------------------------------------------------------------------
# GIK AML Device Project
# boot.py runs once whenever Nano is powered up for the first time/reset
# Please put the initialisation and import modules in this file
# For main loop logic, put the code in main.py
#---------------------------------------------------------------------------------------

import ubluepy
from ubluepy import Peripheral, Service, Characteristic, UUID
import time
import imu
import struct
from machine import I2C, Pin

# IMU setup
bus = I2C(1, scl=Pin(15), sda=Pin(14))
m = imu.IMU(bus)

led = Pin(13, Pin.OUT)
