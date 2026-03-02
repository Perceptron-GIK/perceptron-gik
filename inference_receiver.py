import asyncio, struct, time, os, sys, yaml, torch
import numpy as np
from typing import Callable, Any
from bleak import BleakScanner, BleakClient
from collections import deque

from src.Constants.char_to_key import NUM_CLASSES, INDEX_TO_CHAR
from inference_preprocessing import preprocess
from ml.models.gik_model import create_model

## ================================================== ##
# BLUETOOTH CONFIGURATION

DEVICE_NAME_L = "GIK_Nano_L"
DEVICE_NAME_R = "GIK_Nano_R"

UUID_TX_L = "00001235-0000-1000-8000-00805f9b34fb" 
UUID_TX_R = "00001237-0000-1000-8000-00805f9b34fb" 

PACKER_DTYPE_DEF = "<I" +"f"*6 + ("f"*6 + "B")*5  
assert struct.calcsize(PACKER_DTYPE_DEF) == 153 

DEVICE_SEARCH_RATE = 2.0
RECEIVE_RATE = 30.0
BLE_RECONNECT_DELAY = 1.0     
BLE_MAX_RETRIES = 10
FLUSH_SIZE = int(RECEIVE_RATE*2)
MAX_QUEUE_SIZE = int(RECEIVE_RATE*10)

UNPACKER = struct.Struct(PACKER_DTYPE_DEF)
assert UNPACKER.size == 153

## ================================================== ##
# TRAINING AND INFERENCE CONFIGURATION

PROJECT_ROOT = os.path.dirname(os.path.abspath('__file__'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRAINING_CONFIG_PATH = os.path.join(PROJECT_ROOT, "train_config.yaml")
with open(TRAINING_CONFIG_PATH, "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

EXPERIMENT_MODE = config_data["experiment"]["mode"]
MODE_CONFIG = config_data["modes"][EXPERIMENT_MODE]

TRAINING_CONFIG = {
    **config_data["model"],
    **MODE_CONFIG    
}
TRAINING_CONFIG["output_logits"] = NUM_CLASSES

INFERENCE_CONFIG = {
    "max_seq_length": 10,
    "normalize": True,
    "apply_filtering": True,
    "reduce_dim": True,
    "dim_red_method": "pca", # Set to None if reduce_dim == False
    "dims_ratio": 0.4 # Set to 0.0 if dims_red_method != "pca"
}

MODEL_PATH = os.path.join(PROJECT_ROOT, "best_model.pt")

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

## ================================================== ##
# BLUETOOTH FUNCTIONS

def handler_closure(queue: asyncio.Queue, side: str) -> Callable[[object, bytes], None]:
    first_sample_id = None
    last_sample_id = None
    
    def handler(sender, data):
        nonlocal first_sample_id, last_sample_id
        
        try:
            if len(data) != 153:
                print(f"Unexpected length from {side} hand: {len(data)}")
                return
            
            received_data = UNPACKER.unpack(data)
            sample_id = int(received_data[0])

            t = time.time()
            
            try:
                queue.put_nowait((received_data, t))
            except asyncio.QueueFull:
                print(f"{side} hand queue full, dropping packet {sample_id}")
            
        except Exception as e:
            print(f"Error in {side} handler: {e}")
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
    return nano

async def connect(device_name, uuid, queue):
    if "_R" in device_name:
        side = "right"
    elif "_L" in device_name:
        side = "left"

    retries = 0
    while True:
        try:
            if retries == 0:
                nano = await wait_for_nano(device_name)
                print(f"Found GIK {side} hand")

            async with BleakClient(nano.address) as client:
                retries = 0
                await client.start_notify(uuid, handler_closure(queue, side))
                print(f"Connected to GIK {side} hand, receiving data...")
                while client.is_connected:
                    await asyncio.sleep(1.0)

            print(f"GIK {side} hand disconnected, attempting reconnection...")
            retries = 0

        except Exception as e:
            retries += 1
            print(f"GIK {side} hand connection error (attempt {retries}): {e}")
            if retries >= BLE_MAX_RETRIES:
                print(f"GIK {side} hand max retries reached - re-scanning...")
                retries = 0  

        await asyncio.sleep(BLE_RECONNECT_DELAY)

## ================================================== ##
# INFERENCE FUNCTIONS

def run_inference(
    left: np.ndarray,
    right: np.ndarray,
    prev_char: Any=None
):
    processed_data = preprocess(
        left_data = left,
        right_data = right,
        prev_char = prev_char,
        max_seq_length = INFERENCE_CONFIG["max_seq_length"],
        normalize = INFERENCE_CONFIG["normalize"],
        apply_filtering = INFERENCE_CONFIG["apply_filtering"],
        apply_dim_reduction = INFERENCE_CONFIG["reduce_dim"],
        dim_red_method = INFERENCE_CONFIG["dim_red_method"],
        dims_ratio = INFERENCE_CONFIG["dims_ratio"],
        root_dir = PROJECT_ROOT
    )

    input_dim = processed_data.shape[2]

    model = create_model(
        model_type = TRAINING_CONFIG["model_type"],
        hidden_dim_inner_model = TRAINING_CONFIG['hidden_dim_inner_model'],
        hidden_dim_classification_head = TRAINING_CONFIG['hidden_dim_classification_head'],
        no_layers_classification_head = TRAINING_CONFIG['num_layers'],
        dropout_inner_layers = TRAINING_CONFIG['dropout'],
        inner_model_kwargs = TRAINING_CONFIG['inner_model_prams'],
        output_logits = TRAINING_CONFIG['output_logits'],
        input_dim = input_dim
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        prediction = model.predict(processed_data).item()

    print(INDEX_TO_CHAR[prediction], end="")
    return prediction
    
async def process_queues(left_queue, right_queue):
    max_window_size = INFERENCE_CONFIG["max_seq_length"]
    left_win = deque(maxlen=max_window_size)
    right_win = deque(maxlen=max_window_size)
    prev_char = None

    while True:
        data_l, t_l = await left_queue.get()
        data_l = np.asarray(data_l, dtype=np.float32)
        data_l = np.concatenate([data_l[:, 1:], [t_l]])
        left_win.append(data_l)

        data_r, t_r = await right_queue.get()
        data_r = np.asarray(data_r, dtype=np.float32)
        data_r = np.concatenate([data_r[:, 1:], [t_r]])
        right_win.append(data_r)

        # To-Do: Add logic to check if FSR data is registered before running inference

        if len(left_win) == max_window_size and len(right_win) == max_window_size:
            prev_char = await asyncio.to_thread(run_inference, left_win, right_win, prev_char)

## ================================================== ##

async def main():
    print(f"Waiting for GIK to appear...")

    data_queue_left = asyncio.Queue(MAX_QUEUE_SIZE)
    data_queue_right = asyncio.Queue(MAX_QUEUE_SIZE)

    await asyncio.gather(connect(DEVICE_NAME_L, UUID_TX_L, data_queue_left), 
                         connect(DEVICE_NAME_R, UUID_TX_R, data_queue_right),
                         process_queues(data_queue_left, data_queue_right))

asyncio.run(main())
