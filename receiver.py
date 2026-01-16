#-----------------------------------------------------------------------------------------------------
# This script recieves the data packet via Bluetooth from the Nano
# TO DO: Add the UUID and unpacking service for Right Hand
# The MTU is able to send 153 bytes in one packet, so all data is sent in one notification
# The script utilises threading to run both left and right hand concurrently using asyncio.gather
#------------------------------------------------------------------------------------------------------
import asyncio, struct, time, threading
from bleak import BleakScanner, BleakClient
from datetime import datetime  

DEVICE_NAME_L = "GIK_Nano_L" # Left hand nano name
DEVICE_NAME_R = "GIK_Nano_R" # Right hand nano name

packet_dtype_def = "<I" + "f"*6 + ("f"*6 + "B")*5  # little-endian
assert struct.calcsize(packet_dtype_def) == 153 # match the packet size



def left_handler(sender, data_left):

    if len(data_left) != 153: # sanity check
        print("Unexpected length from left hand", len(data_left))
        return
    # unpack the packet

    (
        sample_id,
        ax_l_base, ay_l_base, az_l_base,
        gx_l_base, gy_l_base, gz_l_base,

        ax_l_thumb, ay_l_thumb, az_l_thumb,
        gx_l_thumb, gy_l_thumb, gz_l_thumb, f_l_thumb_u8,

        ax_l_index, ay_l_index, az_l_index,
        gx_l_index, gy_l_index, gz_l_index, f_l_index_u8,

        ax_l_middle, ay_l_middle, az_l_middle,
        gx_l_middle, gy_l_middle, gz_l_middle, f_l_middle_u8,

        ax_l_ring, ay_l_ring, az_l_ring,
        gx_l_ring, gy_l_ring, gz_l_ring, f_l_ring_u8,

        ax_l_pinky, ay_l_pinky, az_l_pinky,
        gx_l_pinky, gy_l_pinky, gz_l_pinky, f_l_pinky_u8,
    ) = struct.unpack(packet_dtype_def, data_left)

    # convert to Python bools with the same names
    f_l_thumb  = bool(f_l_thumb_u8)
    f_l_index  = bool(f_l_index_u8)
    f_l_middle = bool(f_l_middle_u8)
    f_l_ring   = bool(f_l_ring_u8)
    f_l_pinky  = bool(f_l_pinky_u8)

    t = time.time()
    ts_string = datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm

    print(
        f"id={sample_id} "
        f"time={ts_string} "
        f"a_l_b=({ax_l_base:.4f},{ay_l_base:.4f},{az_l_base:.4f}) "
        f"g_l_b=({gx_l_base:.4f},{gy_l_base:.4f},{gz_l_base:.4f}) "
        f"a_l_t=({ax_l_thumb:.4f},{ay_l_thumb:.4f},{az_l_thumb:.4f}) "
        f"g_l_t=({gx_l_thumb:.4f},{gy_l_thumb:.4f},{gz_l_thumb:.4f}) "
        f"f_l_t={int(f_l_thumb)} "
        f"a_l_i=({ax_l_index:.4f},{ay_l_index:.4f},{az_l_index:.4f}) "
        f"g_l_i=({gx_l_index:.4f},{gy_l_index:.4f},{gz_l_index:.4f}) "
        f"f_l_i={int(f_l_index)} "
        f"a_l_m=({ax_l_middle:.4f},{ay_l_middle:.4f},{az_l_middle:.4f}) "
        f"g_l_m=({gx_l_middle:.4f},{gy_l_middle:.4f},{gz_l_middle:.4f}) "
        f"f_l_m={int(f_l_middle)} "
        f"a_l_r=({ax_l_ring:.4f},{ay_l_ring:.4f},{az_l_ring:.4f}) "
        f"g_l_r=({gx_l_ring:.4f},{gy_l_ring:.4f},{gz_l_ring:.4f}) "
        f"f_l_r={int(f_l_ring)} "
        f"a_l_p=({ax_l_pinky:.4f},{ay_l_pinky:.4f},{az_l_pinky:.4f}) "
        f"g_l_p=({gx_l_pinky:.4f},{gy_l_pinky:.4f},{gz_l_pinky:.4f}) "
        f"f_l_p={int(f_l_pinky)}"
    )

def right_handler(sender, data_right):

    if len(data_right) != 153: # sanity check
        print("Unexpected length from right hand", len(data_right))
        return
    # unpack the packet

    (
        sample_id,
        ax_r_base, ay_r_base, az_r_base,
        gx_r_base, gy_r_base, gz_r_base,

        ax_r_thumb, ay_r_thumb, az_r_thumb,
        gx_r_thumb, gy_r_thumb, gz_r_thumb, f_r_thumb_u8,

        ax_r_index, ay_r_index, az_r_index,
        gx_r_index, gy_r_index, gz_r_index, f_r_index_u8,

        ax_r_middle, ay_r_middle, az_r_middle,
        gx_r_middle, gy_r_middle, gz_r_middle, f_r_middle_u8,

        ax_r_ring, ay_r_ring, az_r_ring,
        gx_r_ring, gy_r_ring, gz_r_ring, f_r_ring_u8,

        ax_r_pinky, ay_r_pinky, az_r_pinky,
        gx_r_pinky, gy_r_pinky, gz_r_pinky, f_r_pinky_u8,
    ) = struct.unpack(packet_dtype_def, data_right)

    # convert to Python bools with the same names
    f_r_thumb  = bool(f_r_thumb_u8)
    f_r_index  = bool(f_r_index_u8)
    f_r_middle = bool(f_r_middle_u8)
    f_r_ring   = bool(f_r_ring_u8)
    f_r_pinky  = bool(f_r_pinky_u8)

    t = time.time()
    ts_string = datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm

    print(
        f"id={sample_id} "
        f"time={ts_string} "
        f"a_r_b=({ax_r_base:.4f},{ay_r_base:.4f},{az_r_base:.4f}) "
        f"g_r_b=({gx_r_base:.4f},{gy_r_base:.4f},{gz_r_base:.4f}) "
        f"a_r_t=({ax_r_thumb:.4f},{ay_r_thumb:.4f},{az_r_thumb:.4f}) "
        f"g_r_t=({gx_r_thumb:.4f},{gy_r_thumb:.4f},{gz_r_thumb:.4f}) "
        f"f_r_t={int(f_r_thumb)} "
        f"a_r_i=({ax_r_index:.4f},{ay_r_index:.4f},{az_r_index:.4f}) "
        f"g_r_i=({gx_r_index:.4f},{gy_r_index:.4f},{gz_r_index:.4f}) "
        f"f_r_i={int(f_r_index)} "
        f"a_r_m=({ax_r_middle:.4f},{ay_r_middle:.4f},{az_r_middle:.4f}) "
        f"g_r_m=({gx_r_middle:.4f},{gy_r_middle:.4f},{gz_r_middle:.4f}) "
        f"f_r_m={int(f_r_middle)} "
        f"a_r_r=({ax_r_ring:.4f},{ay_r_ring:.4f},{az_r_ring:.4f}) "
        f"g_r_r=({gx_r_ring:.4f},{gy_r_ring:.4f},{gz_r_ring:.4f}) "
        f"f_r_r={int(f_r_ring)} "
        f"a_r_p=({ax_r_pinky:.4f},{ay_r_pinky:.4f},{az_r_pinky:.4f}) "
        f"g_r_p=({gx_r_pinky:.4f},{gy_r_pinky:.4f},{gz_r_pinky:.4f}) "
        f"f_r_p={int(f_r_pinky)}"
    )


async def wait_for_left_nano():
    nano_left = None
    while nano_left is None:
        devices = await BleakScanner.discover()
        for d in devices:
            if d.name == DEVICE_NAME_L:
                nano_left = d
                break
        if nano_left is None:
            await asyncio.sleep(0.5)  # wait 0.5s then scan again
    print("Found GIK Left Hand")
    print(nano_left)
    return nano_left

async def wait_for_right_nano():
    nano_right = None
    while nano_right is None:
        devices = await BleakScanner.discover()
        for e in devices:
            if e.name == DEVICE_NAME_R:
                nano_right = e
                break
        if nano_right is None:
            await asyncio.sleep(0.5)  # wait 0.5s then scan again
    print("Found GIK Right Hand")
    print(nano_right)
    return nano_right


async def left():
    nano_left = await wait_for_left_nano()

    while True:

        async with BleakClient(nano_left.address) as client:
            tx_char_left = None
            print("Connected:", nano_left.address)
            for service in client.services:
                print("Service:", service.uuid)
                for char in service.characteristics:
                # Find the char object with that UUID once, then reuse it, did this to avoid string format not matching
                    if str(char.uuid) == "00001235-0000-1000-8000-00805f9b34fb": # UUID for Left hand tx characteristic
                        tx_char_left = char
                        break
                if tx_char_left:
                    break

            print("Left TX characteristic:", tx_char_left)

            await client.start_notify(tx_char_left, left_handler) # Activate notifications/indications on the characteristic.

            while client.is_connected:
                
                # Match your Nano loop rate (100Hz)
                await asyncio.sleep(0.01)

async def right():
    nano_right = await wait_for_right_nano()

    while True:
        async with BleakClient(nano_right.address) as client:
            tx_char_right = None
            print("Connected:", nano_right.address)
            for service in client.services:
                print("Service:", service.uuid)
                for char in service.characteristics:
                # Find the char object with that UUID once, then reuse it, did this to avoid string format not matching
                    if str(char.uuid) == "00001237-0000-1000-8000-00805f9b34fb": # UUID for Right hand tx characteristic
                        tx_char_right = char
                        break
                if tx_char_right:
                    break

            print("Right TX characteristic:", tx_char_right)

            await client.start_notify(tx_char_right, right_handler) # Activate notifications/indications on the characteristic.

            while client.is_connected:
                
                # Match your Nano loop rate (100Hz)
                await asyncio.sleep(0.01)



async def main():
    print(f"Waiting for GIK to appear...")
    await asyncio.gather(left(), right()) # Run both left and right concurrently, asyncio handles the threading


asyncio.run(main())
