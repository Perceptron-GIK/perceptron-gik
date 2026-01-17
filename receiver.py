#-----------------------------------------------------------------------------------------------------
# This script recieves the data packet via Bluetooth from the Nano
# TO DO: Add the UUID and unpacking service for Right Hand
# The MTU is able to send 153 bytes in one packet, so all data is sent in one notification
# The script utilises threading to run both left and right hand concurrently using asyncio.gather
#------------------------------------------------------------------------------------------------------
import asyncio, struct, time
from bleak import BleakScanner, BleakClient
from datetime import datetime  

DEVICE_NAME_L = "GIK_Nano_L" # Left hand nano name
DEVICE_NAME_R = "GIK_Nano_R" # Right hand nano name

UUID_TX_L = "00001235-0000-1000-8000-00805f9b34fb" # Left hand TX characteristic UUID
UUID_TX_R = "00001237-0000-1000-8000-00805f9b34fb" # Right hand TX characteristic UUID

packet_dtype_def = "<I" + "f"*6 + ("f"*6 + "B")*5  # little-endian
assert struct.calcsize(packet_dtype_def) == 153 # match the packet size



def handler(side, data):

    if len(data) != 153: # sanity check
        print(f"Unexpected length from {side} hand", len(data))
        return
    # unpack the packet

    (
        sample_id,
        ax_base, ay_base, az_base,
        gx_base, gy_base, gz_base,

        ax_thumb, ay_thumb, az_thumb,
        gx_thumb, gy_thumb, gz_thumb, f_thumb_u8,

        ax_index, ay_index, az_index,
        gx_index, gy_index, gz_index, f_index_u8,

        ax_middle, ay_middle, az_middle,
        gx_middle, gy_middle, gz_middle, f_middle_u8,

        ax_ring, ay_ring, az_ring,
        gx_ring, gy_ring, gz_ring, f_ring_u8,

        ax_pinky, ay_pinky, az_pinky,
        gx_pinky, gy_pinky, gz_pinky, f_pinky_u8,
    ) = struct.unpack(packet_dtype_def, data)

    # convert to Python bools with the same names
    f_thumb  = bool(f_thumb_u8)
    f_index  = bool(f_index_u8)
    f_middle = bool(f_middle_u8)
    f_ring   = bool(f_ring_u8)
    f_pinky  = bool(f_pinky_u8)

    t = time.time()
    ts_string = datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm

    print(
        f"id={sample_id},{side} "
        f"time={ts_string} "
        f"a_b=({ax_base:.4f},{ay_base:.4f},{az_base:.4f}) "
        f"g_b=({gx_base:.4f},{gy_base:.4f},{gz_base:.4f}) "
        f"a_t=({ax_thumb:.4f},{ay_thumb:.4f},{az_thumb:.4f}) "
        f"g_t=({gx_thumb:.4f},{gy_thumb:.4f},{gz_thumb:.4f}) "
        f"f_t={int(f_thumb)} "
        f"a_i=({ax_index:.4f},{ay_index:.4f},{az_index:.4f}) "
        f"g_i=({gx_index:.4f},{gy_index:.4f},{gz_index:.4f}) "
        f"f_i={int(f_index)} "
        f"a_m=({ax_middle:.4f},{ay_middle:.4f},{az_middle:.4f}) "
        f"g_m=({gx_middle:.4f},{gy_middle:.4f},{gz_middle:.4f}) "
        f"f_m={int(f_middle)} "
        f"a_r=({ax_ring:.4f},{ay_ring:.4f},{az_ring:.4f}) "
        f"g_r=({gx_ring:.4f},{gy_ring:.4f},{gz_ring:.4f}) "
        f"f_r={int(f_ring)} "
        f"a_p=({ax_pinky:.4f},{ay_pinky:.4f},{az_pinky:.4f}) "
        f"g_p=({gx_pinky:.4f},{gy_pinky:.4f},{gz_pinky:.4f}) "
        f"f_p={int(f_pinky)}"
    )


async def wait_for_nano(device_name):
    nano = None
    while nano is None:
        devices = await BleakScanner.discover()
        for d in devices:
            if d.name == device_name:
                nano = d
                break
        if nano is None:
            await asyncio.sleep(0.5)  # wait 0.5s then scan again
    print(nano)
    return nano


async def connect(device_name, uuid):

    # To distingiush left and right from the device name
    if "_R" in device_name:
        side = "Right"
    elif "_L" in device_name:
        side = "Left"

    nano = await wait_for_nano(device_name)
    print(f"Found GIK {side} Hand")
    while True:

        async with BleakClient(nano.address) as client:
            tx_chareft = None
            print("Connected:", nano.address)
            for service in client.services:
                print("Service:", service.uuid)
                for char in service.characteristics:
                # Find the char object with that UUID once, then reuse it, did this to avoid string format not matching
                    if str(char.uuid) == uuid: # UUID for Left hand tx characteristic
                        tx_char = char
                        break
                if tx_char:
                    break

            print(f"{side} Characteristic:", tx_char)

            await client.start_notify(tx_char, handler(side)) # Activate notifications/indications on the characteristic.

            while client.is_connected:
                
                # Match your Nano loop rate (100Hz)
                await asyncio.sleep(0.01)



async def main():
    print(f"Waiting for GIK to appear...")
    await asyncio.gather(connect(DEVICE_NAME_L, UUID_TX_L), connect(DEVICE_NAME_R, UUID_TX_R)) # Run both left and right concurrently, asyncio handles the threading


asyncio.run(main())
