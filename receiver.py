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

# CONSTANTS 
DEVICE_NAME_L = "GIK_Nano_L" # Left hand nano name
DEVICE_NAME_R = "GIK_Nano_R" # Right hand nano name
KEYBOARD_NAME = "Keyboard"

UUID_TX_L = "00001235-0000-1000-8000-00805f9b34fb" # Left hand TX characteristic UUID
UUID_TX_R = "00001237-0000-1000-8000-00805f9b34fb" # Right hand TX characteristic UUID

PACKER_DTYPE_DEF = "<I" +"f"*6 + ("f"*6 + "B")*5  # little-endian
assert struct.calcsize(PACKER_DTYPE_DEF) == 153 # match the packet size

DEVICE_SEARCH_RATE = 2.0       # Frequency with which Bleak searches for a bluetooth device (ensure its is float)
RECEIVE_RATE       = 100.0     # Frequency of packets being received in hertz (ensure its is float)

OVERRIDE_SESSION_ID = False
RIGHT_SESSION_ID = None 
LEFT_SESSION_ID = None
KEYBOARD_SESSION_ID = None

DATA_HEADER = "sample_id,time_stamp,ax_base,ay_base,az_base,gx_base,gy_base,gz_base,ax_thumb,ay_thumb,az_thumb,gx_thumb,gy_thumb,gz_thumb,f_thumb,ax_index,ay_index,az_index,gx_index,gy_index,gz_index,f_index,ax_middle,ay_middle,az_middle,gx_middle,gy_middle,gz_middle,f_middle,ax_ring,ay_ring,az_ring,gx_ring,gy_ring,gz_ring,f_ring,ax_pinky,ay_pinky,az_pinky,gx_pinky,gy_pinky,gz_pinky,f_pinky"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# INITIALISATION   
data_queue_left = asyncio.Queue()
data_queue_right = asyncio.Queue()
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
            data = await queue.get()
            f.write(f"{data}\n")
            flush_counter += 1
            if flush_counter >= RECEIVE_RATE: # Write to the file every second
                f.flush()
                flush_counter = 0


def csv_writer(queue: asyncio.Queue, side: str):
    """Write data from the queue to a csv file.

    Args:
        queue (asyncio.Queue): Queue either Left or Right from which we want to extract data
        side (str): specifies the arduino (either Left or Right) supplying our data
    """
    data_file = get_session_file(side)
    
    if not os.path.exists(data_file):
        with open(data_file, "w") as f:
            f.write(f"{DATA_HEADER}\n")
            
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


def handler_closure(queue: asyncio.Queue , side: str) -> Callable[[object, bytes], None] : 
    """ THis is a Python closure ( Higher Order function), this is basically a trick to call the function defined inside handler with its 
    argument structure while passing in additional arguments that will be defined only in its scope, like the queue and side.

    Args:
        queue (asyncio.Queue): Queue to which the handler will push incoming bluetooth data into in FIFO style
        side (str): Information on which nano we are receiving data from

    Returns:
        function: handler object
    """
    def handler(sender ,data):
        if len(data) != 153: # sanity check
            print(f"Unexpected length from hand {side}", len(data))
            return
        # unpack the packet
        received_data = list(struct.unpack(PACKER_DTYPE_DEF, data))
        
        # Print to terminal
        print_data(received_data)
    
        # Timestamp data
        t = time.time()
        ts_string = datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
        received_data.insert(1, ts_string)
        
        # Push data into the queue
        queue.put_nowait(','.join(str(x) for x in received_data))

    return handler 


async def wait_for_nano(device_name):
    nano = None
    while nano is None:
        devices = await BleakScanner.discover()
        for d in devices:
            if d.name == device_name:
                nano = d
                break
        if nano is None:
            await asyncio.sleep(1/DEVICE_SEARCH_RATE)  # wait 1/DEVICE_SEARCH_RATE seconds then scan again
    print(nano)
    return nano


async def connect(device_name, uuid, queue):
    global keyboard_started
    
    # To distinguish left and right from the device name
    if "_R" in device_name:
        side = "Right"
    elif "_L" in device_name:
        side = "Left"
        device_name = "Arduino"

    nano = await wait_for_nano(device_name)
    print(f"Found GIK {side} Hand")
    
    # Start keyboard logging on first connection
    if not keyboard_started:
        keyboard_started = True
        keyboard_file = get_session_file(KEYBOARD_NAME)
        asyncio.create_task(start_keyboard(keyboard_file))
        print(f"Started keyboard logging at {keyboard_file}")
    
    # Start csv writers
    csv_writer(queue, side)
    
    # Main bluetooth loop
    while not stop_event.is_set():
        async with BleakClient(nano.address) as client:
            # Upon receiving notification from the nano we call the handler function
            await client.start_notify(uuid, handler_closure(queue, side))
            while client.is_connected and not stop_event.is_set():
                await asyncio.sleep(1/RECEIVE_RATE)


async def main():
    print(f"Waiting for GIK to appear...")
    # Run both left and right hand connections concurrently
    await asyncio.gather(connect(DEVICE_NAME_L, UUID_TX_L, data_queue_left), connect(DEVICE_NAME_R, UUID_TX_R, data_queue_right))

asyncio.run(main())
