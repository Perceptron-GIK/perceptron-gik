import ubluepy
from ubluepy import Peripheral, constants
import machine
import time

# --- Configuration ---
DEVICE_NAME = "Nano-BLE"
LED_PIN = 13
led = machine.Pin(LED_PIN, machine.Pin.OUT)

print("---------------------------------------")
print(f"Starting {DEVICE_NAME}...")

def start_bluetooth():
    # 1. Initialize the Peripheral
    p = Peripheral()
    
    # Start Advertising
    # The 'connectable' flag tells the phone "You can talk to me"
    try:
        p.advertise(device_name=DEVICE_NAME, connectable=True)
        print("SUCCESS: Advertising started.")
        print("Scan with nRF Connect to see the device.")
    except Exception as e:
        print(f"Advertising Failed: {e}")

start_bluetooth()

# --- Main Loop ---
# Blink pattern: Fast double-blink indicates HeartBeat"
while True:
    led.value(1)
    time.sleep(0.1)
    led.value(0)
    time.sleep(0.1)
    led.value(1)
    time.sleep(0.1)
    led.value(0)
    time.sleep(0.7)