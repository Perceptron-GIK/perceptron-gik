#----------------------------------------------------------------------------------------
# GIK AML Device Project
# main.py runs after boot.py and will be executed till the board is disconnected from power supply
# Please put the initialisation and import modules in boot.py
# For main loop logic, put the code in this file
#---------------------------------------------------------------------------------------

# Stop any existing advertising
try:
    p.advertise_stop()
except:
    pass

p = Peripheral()
uart_service = Service(UUID(0x1234))
tx_char = Characteristic(UUID(0x1235))
uart_service.addCharacteristic(tx_char)
p.addService(uart_service)
p.advertise(device_name="T", connectable=True)
sample_id = 0

print("IMU streamer ready - enable TX notifications")

def send_imu():
  global sample_id
  ax, ay, az = m.accel()
  gx, gy, gz = m.gyro()
  packet1 = struct.pack('<Ifff', sample_id,ax,ay,az)  # 23 bytes max per packet (we use 22 here)
  packet2 = struct.pack('<Ifff', sample_id,gx,gy,gz) # second packet with the rest data
  tx_char.write(packet1)
  tx_char.write(packet2)
  #print(m.accel(), m.gyro())

  sample_id = (sample_id + 1) & 0xFFFFFFFF # Increase sample_id by 1 after one round of sending (2 bytes)

while True:
    led.value(1)
    send_imu()
    led.value(0)
    time.sleep_ms(25)