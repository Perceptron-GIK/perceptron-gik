#----------------------------------------------------------------------------------------
# GIK AML Device Project
# main.py runs after boot.py and will be executed till the board is disconnected from power supply
# Please put the initialisation and import modules in boot.py
# For main loop logic, put the code in this file
#---------------------------------------------------------------------------------------

p = Peripheral()
uart_service = Service(UUID(0x1234))
tx_char = Characteristic(UUID(0x1235))
uart_service.addCharacteristic(tx_char)
p.addService(uart_service)
p.advertise(device_name="T", connectable=True)

print("IMU streamer ready - enable TX notifications")

def send_imu():
  ax, ay, az = m.accel()
  msg = f"{ax:.2f},{ay:.2f},{az:.2f}"
  tx_char.write(msg.encode())
  print(m.accel())

while True:
    led.value(1)
    send_imu()
    led.value(0)
    time.sleep_ms(50)