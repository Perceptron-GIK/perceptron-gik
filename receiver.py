#-----------------------------------------------------------------------------------------------------
# This script recieves the data packet via Bluetooth from the Nano
# TO DO: Add the UUID and unpacking service for Right Hand
# The MTU is able to send 153 bytes in one packet, so all data is sent in one notification
# The script utilises threading to run both left and right hand concurrently using asyncio.gather
#------------------------------------------------------------------------------------------------------
import asyncio, struct, time, os
from typing import Callable
from bleak import BleakScanner, BleakClient
from datetime import datetime  
from src.keyboard.keyboard_ext import start_keyboard, stop_event
import cProfile

# CONSTANTS 
DEVICE_NAME_L = "GIK_Nano_L" # Left hand nano name
DEVICE_NAME_R = "GIK_Nano_R" # Right hand nano name
KEYBOARD_NAME = "Keyboard"

UUID_TX_L = "00001235-0000-1000-8000-00805f9b34fb" # Left hand TX characteristic UUID
UUID_TX_R = "00001237-0000-1000-8000-00805f9b34fb" # Right hand TX characteristic UUID

PACKER_DTYPE_DEF = "<I" +"f"*6 + ("f"*6 + "B")*5  # little-endian
assert struct.calcsize(PACKER_DTYPE_DEF) == 153 # match the packet size

DEVICE_SEARCH_RATE = 2.0       # Frequency with which Bleak searches for a bluetooth device (ensure its is float)
RECEIVE_RATE       = 30.0     # Frequency of packets being received in hertz (ensure its is float)
BLE_RECONNECT_DELAY = 1.0     # Seconds to wait before attempting BLE reconnection
BLE_MAX_RETRIES     = 10       # Max consecutive reconnection attempts before re-scanning
FLUSH_SIZE         = int(RECEIVE_RATE*2)  # Number of packets to receive before flushing to disk (e.g., every second)
MAX_QUEUE_SIZE = int(RECEIVE_RATE *10)

OVERRIDE_SESSION_ID = False
RIGHT_SESSION_ID = None 
LEFT_SESSION_ID = None
KEYBOARD_SESSION_ID = 1

# Add this near the top with other constants
UNPACKER = struct.Struct(PACKER_DTYPE_DEF)
assert UNPACKER.size == 153


DATA_HEADER = "sample_id,ax_base,ay_base,az_base,gx_base,gy_base,gz_base,ax_thumb,ay_thumb,az_thumb,gx_thumb,gy_thumb,gz_thumb,f_thumb,ax_index,ay_index,az_index,gx_index,gy_index,gz_index,f_index,ax_middle,ay_middle,az_middle,gx_middle,gy_middle,gz_middle,f_middle,ax_ring,ay_ring,az_ring,gx_ring,gy_ring,gz_ring,f_ring,ax_pinky,ay_pinky,az_pinky,gx_pinky,gy_pinky,gz_pinky,f_pinky,time_stamp"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# INITIALISATION   

keyboard_started = False


def get_session_file(name: str) -> str:
    """Get a CSV file path with session ID
    
    Creates data directory if needed, manages session metadata, and returns
    the path for the new session's CSV file.
    
    Args:
        name: Identifier for the data source (e.g., 'Left', 'Right', 'Keyboard')
    
    Returns:
        Path to the CSV file for this session (e.g., 'data/Left_1.csv')
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    metadata_file = os.path.join(DATA_DIR, f"metadata_{name}.txt")
    session_id = None
    
    # Handle override for Left/Right
    if OVERRIDE_SESSION_ID :
        if name == "Left":
            session_id = LEFT_SESSION_ID 
        elif name == "Right" :
            session_id = RIGHT_SESSION_ID
        else :
            session_id = KEYBOARD_SESSION_ID
            
    elif os.path.exists(metadata_file):
        with open(metadata_file, "r") as m:
            session_id = int(m.readline().rstrip()) + 1
        with open(metadata_file, "w") as m:
            m.write(f"{session_id}\n")
    else:
        with open(metadata_file, "w") as m:
            m.write("1\n")
            session_id = 1
    
    return os.path.join(DATA_DIR, f"{name}_{session_id}.csv")


def _prepare_csv_file(side: str) -> str:
    """Pre-initialise the CSV file and return the file path.

    Creates the session file and writes the header before any BLE data
    arrives so that initial samples are not lost.

    Args:
        side: 'Left' or 'Right'

    Returns:
        Path to the ready-to-append CSV file.
    """
    data_file = get_session_file(side)
    if not os.path.exists(data_file):
        with open(data_file, "w") as f:
            f.write(f"{DATA_HEADER}\n")
    return data_file


async def _csv_writer(queue: asyncio.Queue, file_name: str ):
    """ Async function that takes data out of the write queue and writes it a given ssv file

    Args:
        queue (asyncio.Queue): Queue either Left or Right from which we want to extract data
        file_name (str): csv file to which we want to write data
    """
    with open(file_name, 'a') as f:
        flush_counter = 0
        
        # Main writer logic loop
        while True:
            data,t = await queue.get()
            data = list(data) # Convert to list to use join
            t = str(t) # convert from float() to string
            data.append(t)

            # Convert to CSV format
            data = ','.join(str(x) for x in data)
            f.write(f"{data}\n")
            flush_counter += 1
            if flush_counter >= (FLUSH_SIZE): # Write to the file every second
                f.flush()
                flush_counter = 0



def csv_writer(queue: asyncio.Queue, data_file: str):
    """Start the async CSV writer task for an already-prepared file.

    Args:
        queue (asyncio.Queue): Queue either Left or Right from which we want to extract data
        data_file (str): path to the CSV file (already created with header)
    """
    asyncio.create_task(_csv_writer(queue, data_file))


def print_data(bluetooth_data: list) -> None:
    """Print data to console
    """
    
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
    ) = bluetooth_data
    
    # convert to Python bools with the same names
    f_thumb  = bool(f_thumb_u8)
    f_index  = bool(f_index_u8)
    f_middle = bool(f_middle_u8)
    f_ring   = bool(f_ring_u8)
    f_pinky  = bool(f_pinky_u8)
    
    print(
            f"id={sample_id}"
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


def handler_closure(queue: asyncio.Queue, side: str) -> Callable[[object, bytes], None]:
    first_sample_id = None
    last_sample_id = None
    packet_count = 0
    gap_count = 0  # Track total missed packets
    handler_times = []  # Track handler performance
    
    def handler(sender, data):
        nonlocal first_sample_id, last_sample_id, packet_count, gap_count
        
        # start_time = time.perf_counter()  # Start timing (for debugging uncomment to run)
        
        try:
            if len(data) != 153:
                print(f"Unexpected length from hand {side}: {len(data)}")
                return
            
            received_data = UNPACKER.unpack(data)
            sample_id = int(received_data[0])

            t = time.time()
            
            try:
                queue.put_nowait((received_data, t))
            except asyncio.QueueFull:
                print(f"{side} Hand queue full, dropping packet {sample_id}")
            
            # For debugging and dropout checking (uncomment to run)
            
            # # Track first packet received
            # if first_sample_id is None:
            #     first_sample_id = sample_id
            #     print(f"\n{side} Hand first packet sample_id={sample_id}\n")
            
            # # Count gaps 
            # if last_sample_id is not None:
            #     expected = last_sample_id + 1
            #     if sample_id != expected:
            #         gap_count += (sample_id - expected)
            
            # last_sample_id = sample_id
            # packet_count += 1
            
            # elapsed_us = (time.perf_counter() - start_time) * 1_000_000  # microseconds
            # handler_times.append(elapsed_us)
            
            # # Print summary every 250 packets
            # if packet_count % 250 == 0:
            #     total_sent = packet_count + gap_count
            #     loss_rate = (gap_count / total_sent) * 100 if total_sent > 0 else 0
                
            #     # Calculate handler performance
            #     avg_time = sum(handler_times) / len(handler_times)
            #     p95_time = sorted(handler_times)[int(len(handler_times) * 0.95)]
            #     max_time = max(handler_times)
                
            #     print(f"{side}: {packet_count} received, {gap_count} missed, {loss_rate:.2f}% loss")
            #     print(f"{side}: Handler avg={avg_time:.0f}µs, p95={p95_time:.0f}µs, max={max_time:.0f}µs")
                
            #     handler_times.clear()
                
        except Exception as e:
            print(f"ERROR in {side} handler: {e}")
            import traceback
            traceback.print_exc()
    
    return handler 


async def wait_for_nano(device_name):
    nano = None
    while nano is None:
        nano = await BleakScanner.find_device_by_filter(
            lambda d, ad: d.name == device_name,
            timeout=1/DEVICE_SEARCH_RATE
        )
    print(nano)
    return nano


async def connect(device_name, uuid, queue):
    global keyboard_started
    
    # To distinguish left and right from the device name
    if "_R" in device_name:
        side = "Right"
    elif "_L" in device_name:
        side = "Left"

    # Pre-initialise CSV file before BLE connection
    data_file = _prepare_csv_file(side)

    # Start keyboard logging on first connection
    if not keyboard_started:
        keyboard_started = True
        keyboard_file = get_session_file(KEYBOARD_NAME)
        asyncio.create_task(start_keyboard(keyboard_file))
        print(f"Started keyboard logging at {keyboard_file}")
    
    # Start csv writer task with the already-prepared file
    csv_writer(queue, data_file)
    await asyncio.sleep(0) # Here to make sure the CSV writer task starts before BLE

    # Reconnection loop
    retries = 0
    while not stop_event.is_set():
        try:
            # Scan for the device (re-scan if we exhausted retries)
            if retries == 0:
                nano = await wait_for_nano(device_name)
                print(f"Found GIK {side} Hand")

            async with BleakClient(nano.address) as client:
                retries = 0
                # Upon receiving notification from the nano we call the handler function
                await client.start_notify(uuid, handler_closure(queue, side))
                print(f"Connected to GIK {side} Hand - receiving data")
                while client.is_connected and not stop_event.is_set():
                    await asyncio.sleep(1.0) # Keep the connection and dont need to be too fast (free up CPU)

            # To address the Nanos disconnect suddenly without able to reconnet 
            print(f"GIK {side} Hand disconnected - attempting reconnection...")
            retries = 0
        except Exception as e:
            retries += 1
            print(f"GIK {side} Hand connection error (attempt {retries}): {e}")
            if retries >= BLE_MAX_RETRIES:
                print(f"GIK {side} Hand max retries reached - re-scanning...")
                retries = 0  

        await asyncio.sleep(BLE_RECONNECT_DELAY)


async def main():
    print(f"Waiting for GIK to appear...")

    data_queue_left = asyncio.Queue(MAX_QUEUE_SIZE)
    data_queue_right = asyncio.Queue(MAX_QUEUE_SIZE)

    # Run both left and right hand connections concurrently
    await asyncio.gather(connect(DEVICE_NAME_L, UUID_TX_L, data_queue_left), connect(DEVICE_NAME_R, UUID_TX_R, data_queue_right))

#cProfile.run('asyncio.run(main())') # Run with profiler
asyncio.run(main())

