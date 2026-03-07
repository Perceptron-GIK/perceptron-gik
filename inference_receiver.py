import asyncio, struct, time, os, sys, yaml, torch
import numpy as np
from typing import Callable, Any
from bleak import BleakScanner, BleakClient

from src.Constants.char_to_key import NUM_CLASSES, INDEX_TO_CHAR, CHAR_TO_INDEX, FULL_COORDS
from src.inference.sliding_window import SlidingWindow
from src.inference.autocorrect import AutoCorrector
from src.visualisation.visualisation import get_closest_coordinate
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

TRAINING_DATA_DIR = config_data["data"]["data_dir"]
PROCESSED_TRAINING_DATA = os.path.join(TRAINING_DATA_DIR, "processed_dataset.pt")

EXPERIMENT_MODE = config_data["experiment"]["mode"]
MODE_CONFIG = config_data["modes"][EXPERIMENT_MODE]

TRAINING_CONFIG = {
    **config_data["model"],
    **MODE_CONFIG    
}
TRAINING_CONFIG["output_logits"] = NUM_CLASSES

INFERENCE_CONFIG = {
    "max_seq_length": 19,
    "normalize": False,
    "apply_filtering": False,
    "reduce_dim": False,
    "dim_red_method": "pca", # Set to None if reduce_dim == False
    "dims_ratio": 0.5 # Set to 0.0 if dims_red_method != "pca"
}

MODEL_PATH = os.path.join(PROJECT_ROOT, "best_model.pt")

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

FSR_INDICES = [12, 19, 26, 33, 40]

# Valid checkers: "pyspell", "neuspell-bert"
AUTOCORRECTOR = AutoCorrector(checker_type="pyspell", max_len=10)

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
    left: np.ndarray=None,
    right: np.ndarray=None,
    left_pointer: int=None,
    right_pointer: int=None,
    prev_char: Any=None
):
    processed_data = preprocess(
        left_data = left,
        right_data = right,
        left_pointer = left_pointer,
        right_pointer = right_pointer,
        prev_char = prev_char,
        mode = EXPERIMENT_MODE,
        max_seq_length = INFERENCE_CONFIG["max_seq_length"],
        normalize = INFERENCE_CONFIG["normalize"],
        apply_filtering = INFERENCE_CONFIG["apply_filtering"],
        apply_dim_reduction = INFERENCE_CONFIG["reduce_dim"],
        dim_red_method = INFERENCE_CONFIG["dim_red_method"],
        dims_ratio = INFERENCE_CONFIG["dims_ratio"],
        root_dir = PROJECT_ROOT,
        training_dataset = PROCESSED_TRAINING_DATA
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

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        if EXPERIMENT_MODE == "classification":
            predicted_idx = model.predict(processed_data).item()
        else:
            predicted_idx = CHAR_TO_INDEX[get_closest_coordinate(model.predict_coords(processed_data), FULL_COORDS)]

    predicted_char = INDEX_TO_CHAR[predicted_idx]
    AUTOCORRECTOR.process_char(predicted_char)
    sys.stdout.write(predicted_char)
    sys.stdout.flush()
    
    return predicted_idx
    
async def process_queues(left_queue, right_queue):
    left_win = SlidingWindow()
    right_win = SlidingWindow()
    left_all = SlidingWindow(maxlen=1000)
    right_all = SlidingWindow(maxlen=1000)
    prev_char = None

    while True:
        left_task = asyncio.create_task(left_queue.get())
        right_task = asyncio.create_task(right_queue.get())
        completed, _ = await asyncio.wait(
            [left_task, right_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in completed:
            data, t = task.result()
            data = np.asarray(data, dtype=np.float32)
            data = np.concatenate([data[1:], [t]])

            if task == left_task:
                left_win.append(data)      
                left_task = asyncio.create_task(left_queue.get())
                triggered_hand = "left"

            elif task == right_task:
                right_win.append(data)
                right_task = asyncio.create_task(right_queue.get())
                triggered_hand = "right"
        
        idx = left_win.fsr_detected(fsr_indices=FSR_INDICES) if triggered_hand == "left" else right_win.fsr_detected(fsr_indices=FSR_INDICES)
        if idx is None:
            continue
        chunk = np.stack(left_win.pop_chunk(idx+1)) if triggered_hand == "left" else np.stack(right_win.pop_chunk(idx+1))
        if chunk.shape[0] <= 2:
            continue
        pointer = chunk.shape[0]
        timestamp = chunk[-1][-1]
        opp_idx = right_win.timestamp_matched(timestamp=timestamp) if triggered_hand == "left" else left_win.timestamp_matched(timestamp=timestamp)
        if opp_idx:
            opp_chunk = np.stack(right_win.pop_chunk(opp_idx+1)) if triggered_hand == "left" else np.stack(left_win.pop_chunk(opp_idx+1))
        else:
            opp_chunk = np.zeros_like(chunk)
        opp_pointer = opp_chunk.shape[0]
        
        if triggered_hand == "left":
            left_all.append(chunk)
            right_all.append(opp_chunk)
            # prev_char = await asyncio.to_thread(run_inference, np.vstack(tuple(left_all.data)), np.vstack(tuple(right_all.data)), pointer, opp_pointer, prev_char)
            prev_char = await asyncio.to_thread(run_inference, chunk, opp_chunk, None, None, prev_char)
        else:
            left_all.append(opp_chunk)
            right_all.append(chunk)
            # prev_char = await asyncio.to_thread(run_inference, np.vstack(tuple(left_all.data)), np.vstack(tuple(right_all.data)), opp_pointer, pointer, prev_char)
            prev_char = await asyncio.to_thread(run_inference, opp_chunk, chunk, None, None, prev_char)

## ================================================== ##

async def main():
    print("Waiting for GIK to appear...")

    data_queue_left = asyncio.Queue(MAX_QUEUE_SIZE)
    data_queue_right = asyncio.Queue(MAX_QUEUE_SIZE)

    await asyncio.gather(connect(DEVICE_NAME_L, UUID_TX_L, data_queue_left), 
                         connect(DEVICE_NAME_R, UUID_TX_R, data_queue_right),
                         process_queues(data_queue_left, data_queue_right))

asyncio.run(main())
