# This script recieves the data packet via Bluetooth 
import asyncio, struct, time
from collections import defaultdict
from bleak import BleakScanner, BleakClient
from datetime import datetime  

DEVICE_NAME = "T"

pending = defaultdict(dict)  # sample_id -> {'t': ..., 'acc': (...), 'gyro': (...)}

# def handler(sender, data):
#     # decode packet format 
#     sample_id, v1, v2, v3 = struct.unpack("<Ifff", data)

#     # choose type based on arrival order (first=facc, second=fgyro)
#     slot = pending[sample_id]
#     if 't' not in slot:
#         slot['t'] = time.time()  # one timestamp for the whole sample
#     if 'acc' not in slot:
#         slot['acc'] = (v1, v2, v3)
#     else:
#         slot['gyro'] = (v1, v2, v3)

#     if 'acc' in slot and 'gyro' in slot:
#         t  = slot['t']
#         ax, ay, az = slot['acc']
#         gx, gy, gz = slot['gyro']
#         print(f"{sample_id} {t:.3f}  acc={ax:.2f},{ay:.2f},{az:.2f}  gyro={gx:.2f},{gy:.2f},{gz:.2f}")
#         writer.writerow([sample_id, t, ax, ay, az, gx, gy, gz])
#         del pending[sample_id]

async def wait_for_nano():
    print(f"Waiting for GIK to appear...")
    nano = None
    while nano is None:
        devices = await BleakScanner.discover()
        for d in devices:
            if d.name == DEVICE_NAME:
                nano = d
                break
        if nano is None:
            await asyncio.sleep(2)  # wait 2s then scan again
    print("Found GIK Left Hand")
    return nano

async def main():
    nano = await wait_for_nano()

    async with BleakClient(nano.address) as client:
        tx_char = None
        print("Connected:", nano.address)
        for service in client.services:
            print("Service:", service.uuid)
            for char in service.characteristics:
            # Find the char object with that UUID once, then reuse it
                if str(char.uuid) == "00001235-0000-1000-8000-00805f9b34fb":
                    tx_char = char
                    break
            if tx_char:
                break

        print("Resolved TX characteristic:", tx_char)

        while True:
            t = time.time()
            ts_string = datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
            # Poll latest packet from Nano
            data = await client.read_gatt_char(tx_char)
            #print("RAW:", data, "len:", len(data))
            # Decode one packet: sample_id + 3 floats
            sample_id, v1, v2, v3 = struct.unpack("<Ifff", data)
            print(f"Recieved from GIK Left: id={sample_id} t={ts_string} v1={v1:.4f} v2={v2:.4f} v3={v3:.4f}")

            # Match your Nano loop rate (e.g. 50 ms)
            #await asyncio.sleep(0.05)

asyncio.run(main())
