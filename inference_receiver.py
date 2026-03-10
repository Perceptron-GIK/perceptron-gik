import asyncio, struct, time, os, sys, yaml, torch
import numpy as np
from typing import Callable, Any, Optional, Dict, List
from bleak import BleakScanner, BleakClient

from src.Constants.char_to_key import NUM_CLASSES, INDEX_TO_CHAR, CHAR_TO_INDEX, FULL_COORDS
from src.inference.sliding_window import SlidingWindow
from src.inference.autocorrect import AutoCorrector
from src.visualisation.visualisation import get_closest_coordinate
from src.decoding.lm_fusion import (
    build_char_ngram_lm,
    build_interpolated_char_lm,
    fuse_single_step_logits_with_lm,
    get_logits_single_tta,
)
from inference_preprocessing import preprocess
from ml.models.gik_model import create_model

## ================================================== ##
# BLUETOOTH CONFIGURATION

DEVICE_NAME_L = "GIK_Nano_L"
DEVICE_NAME_R = "GIK_Nano_R"

# Service UUIDs from GIK_Hand_Config.h - discovery by UUID (Mac may cache name as "Arduino")
SERVICE_UUID_L = "00001234-0000-1000-8000-00805f9b34fb"  # Left: ServiceID 1234
SERVICE_UUID_R = "00001236-0000-1000-8000-00805f9b34fb"  # Right: ServiceID 1236
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

# Config path: use env var, or train_config.yaml
_DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "train_config.yaml")
TRAINING_CONFIG_PATH = os.environ.get("GIK_TRAIN_CONFIG", _DEFAULT_CONFIG)
if not os.path.isfile(TRAINING_CONFIG_PATH):
    TRAINING_CONFIG_PATH = _DEFAULT_CONFIG
with open(TRAINING_CONFIG_PATH, "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

TRAINING_DATA_DIR = config_data["data"]["data_dir"]
PROCESSED_TRAINING_DATA = os.path.join(TRAINING_DATA_DIR, "processed_dataset.pt")

EXPERIMENT = config_data["experiment"]
EXPERIMENT_MODE = EXPERIMENT["mode"]
MODE_CONFIG = config_data["modes"][EXPERIMENT_MODE]

# Resolve output_logits from mode config (e.g. "NUM_CLASSES" -> actual value)
OUTPUT_LOGITS_REGISTRY = {"NUM_CLASSES": NUM_CLASSES}
_output_logits = MODE_CONFIG.get("output_logits", "NUM_CLASSES")
if isinstance(_output_logits, str):
    _output_logits = OUTPUT_LOGITS_REGISTRY.get(_output_logits, NUM_CLASSES)

TRAINING_CONFIG = {
    **config_data["model"],
    **MODE_CONFIG,
    "output_logits": _output_logits,
}

# Inference config derived from experiment (must match training preprocessing)
DIM_RED_CFG = EXPERIMENT.get("dim_reduction", {})
NORMALIZE_CFG = EXPERIMENT.get("normalize", True)
NORMALIZE_ENABLED = NORMALIZE_CFG.get("enabled", True) if isinstance(NORMALIZE_CFG, dict) else bool(NORMALIZE_CFG)

INFERENCE_CONFIG = {
    "max_seq_length": EXPERIMENT.get("max_seq_length", 19),
    "normalize": NORMALIZE_ENABLED,
    "apply_filtering": EXPERIMENT.get("apply_filtering", False),
    "reduce_dim": DIM_RED_CFG.get("enabled", False),
    "dim_red_method": DIM_RED_CFG.get("method", "pca"),
    "dims_ratio": DIM_RED_CFG.get("pca", {}).get("dims_ratio", 0.5),
    "append_prev_char": config_data["experiment"].get("append_prev_char_feature", True),
    "alignment_prev_windows": int(config_data["experiment"].get("alignment_context", {}).get("prev_windows", 0)),
    "alignment_future_windows": int(config_data["experiment"].get("alignment_context", {}).get("future_windows", 0)),
}

TRAIN_CFG = config_data.get("train", {})

# Simple models (CNN/RNN/LSTM) - optional alternative to GIK model for classification
SIMPLE_MODEL_CFG = config_data.get("simple_model", {})
SIMPLE_MODEL_ENABLED = bool(SIMPLE_MODEL_CFG.get("enabled", False)) and EXPERIMENT_MODE == "classification"
SIMPLE_MODEL_PATH = os.path.join(PROJECT_ROOT, SIMPLE_MODEL_CFG.get("path", "models_trained/cnn_classifier.pt"))
SIMPLE_MODEL_TYPE = SIMPLE_MODEL_CFG.get("type", "cnn")

# LM fusion for real-time inference (classification only)
LM_FUSION_ENABLED = EXPERIMENT_MODE == "classification" and TRAIN_CFG.get("lm_fusion_inference", False)
LM_INFERENCE_BETA = float(TRAIN_CFG.get("lm_inference_beta", 0.4))
LM_ORDER = int(TRAIN_CFG.get("lm_order", 3))
LM_HISTORY_LEN = max(0, LM_ORDER - 1)
LM_ADD_K = float(TRAIN_CFG.get("lm_add_k", 0.05))
LM_USE_INTERPOLATED = bool(TRAIN_CFG.get("lm_use_interpolated", True))
TTA_PASSES = max(1, int(TRAIN_CFG.get("lm_tta_passes", 1)))
TTA_NOISE_STD = float(TRAIN_CFG.get("lm_tta_noise_std", 0.0))
TTA_SCALE_JITTER = float(TRAIN_CFG.get("lm_tta_scale_jitter", 0.0))

if LM_FUSION_ENABLED:
    print(f"LM fusion enabled: beta={LM_INFERENCE_BETA}, order={LM_ORDER}, interpolated={LM_USE_INTERPOLATED}", file=sys.stderr)

_INFERENCE_LM: Optional[dict] = None
_INFERENCE_MODEL: Optional[torch.nn.Module] = None

MODEL_PATH = os.path.join(PROJECT_ROOT, TRAIN_CFG.get("model_save_path", "gik_model_OP_.pt"))

# MPS (Mac) produces different numerical results than CUDA; can cause model to collapse to one class.
# Force CPU on Mac for consistent predictions (matches CUDA-trained model). Set GIK_USE_MPS=1 to use MPS.
_USE_MPS = os.environ.get("GIK_USE_MPS", "").lower() in ("1", "true", "yes")
if torch.cuda.is_available():
    DEVICE = "cuda"
elif _USE_MPS and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
    if torch.backends.mps.is_available():
        print("Using CPU for inference (MPS disabled; set GIK_USE_MPS=1 to use MPS)", file=sys.stderr)

FSR_INDICES = [12, 19, 26, 33, 40]

AUTOCORRECTOR = AutoCorrector(checker_type="pyspell", max_len=10)

inference_predictions = ""

## ================================================== ##
# BLUETOOTH FUNCTIONS

def handler_closure(queue: asyncio.Queue, side: str) -> Callable[[object, bytes], None]:
    first_sample_id = None
    last_sample_id = None
    
    def handler(sender, data):
        nonlocal first_sample_id, last_sample_id
        
        try:
            if len(data) != 153:
                print(f"Unexpected length from {side} hand: {len(data)}", file=sys.stderr)
                return
            
            received_data = UNPACKER.unpack(data)
            sample_id = int(received_data[0])

            t = time.time()
            
            try:
                queue.put_nowait((received_data, t))
            except asyncio.QueueFull:
                print(f"{side} hand queue full, dropping packet {sample_id}", file=sys.stderr)
            
        except Exception as e:
            print(f"Error in {side} handler: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    return handler 

def _match_gik_device(service_uuid):
    """Match by service UUID. Macs may cache 'Arduino' instead of GIK_Nano_L/R."""
    def _match(d, ad):
        uuids = ad.service_uuids or set()
        return service_uuid in uuids
    return _match


async def wait_for_nano(service_uuid):
    nano = None
    while nano is None:
        nano = await BleakScanner.find_device_by_filter(
            _match_gik_device(service_uuid),
            timeout=1/DEVICE_SEARCH_RATE
        )
    return nano

async def connect(device_name, service_uuid, char_uuid, queue):
    if "_R" in device_name:
        side = "right"
    elif "_L" in device_name:
        side = "left"

    retries = 0
    while True:
        try:
            if retries == 0:
                nano = await wait_for_nano(service_uuid)
                svc = service_uuid[4:8] if len(service_uuid) >= 8 else "?"
                print(f"\nFound GIK {side} hand (device name: {nano.name!r}, service {svc})", file=sys.stderr)

            async with BleakClient(nano.address) as client:
                retries = 0
                await client.start_notify(char_uuid, handler_closure(queue, side))
                print(f"\nConnected to GIK {side} hand, receiving data...", file=sys.stderr)
                while client.is_connected:
                    await asyncio.sleep(1.0)

            print(f"\nGIK {side} hand disconnected, attempting reconnection...", file=sys.stderr)
            retries = 0

        except Exception as e:
            retries += 1
            print(f"\nGIK {side} hand connection error (attempt {retries}): {e}", file=sys.stderr)
            if retries >= BLE_MAX_RETRIES:
                print(f"GIK {side} hand max retries reached - re-scanning...", file=sys.stderr)
                retries = 0  

        await asyncio.sleep(BLE_RECONNECT_DELAY)

## ================================================== ##
# INFERENCE FUNCTIONS

def _get_inference_lm() -> Optional[dict]:
    """Build LM from training labels (cached)."""
    global _INFERENCE_LM
    if _INFERENCE_LM is not None:
        return _INFERENCE_LM
    if not LM_FUSION_ENABLED:
        return None
    try:
        data = torch.load(PROCESSED_TRAINING_DATA, weights_only=False)
        labels = data.get("labels", [])
        if not isinstance(labels, list) or len(labels) == 0:
            return None
        if LM_USE_INTERPOLATED and LM_ORDER > 1:
            _INFERENCE_LM = build_interpolated_char_lm(labels, max_order=LM_ORDER, add_k=LM_ADD_K)
        else:
            _INFERENCE_LM = build_char_ngram_lm(labels, order=LM_ORDER, add_k=LM_ADD_K)
        return _INFERENCE_LM
    except Exception as e:
        print(f"LM build failed: {e}, disabling LM fusion", file=sys.stderr)
        return None

def _get_simple_model_input_dim(model) -> Optional[int]:
    """Infer expected input feature dim from simple model architecture."""
    if hasattr(model, "convs") and model.convs:
        return model.convs[0][0].in_channels
    if hasattr(model, "rnn"):
        return model.rnn.weight_ih_l0.shape[1]
    if hasattr(model, "lstm"):
        return model.lstm.weight_ih_l0.shape[1]
    if hasattr(model, "resblock1"):
        return model.resblock1.conv1.in_channels
    return None


def _get_inference_model(input_dim: int):
    """Load and cache model (reused across inferences). Uses simple model if enabled."""
    global _INFERENCE_MODEL
    if _INFERENCE_MODEL is not None and getattr(_INFERENCE_MODEL, "_gik_input_dim", None) == input_dim:
        return _INFERENCE_MODEL

    if SIMPLE_MODEL_ENABLED:
        from simple_models import (
            SimpleCNNClassifier,
            SimpleRNNClassifier,
            SimpleLSTMClassifier,
            SimpleCNNLSTMClassifier,
            load_classifier_checkpoint,
        )
        model_cls = {
            "cnn": SimpleCNNClassifier,
            "rnn": SimpleRNNClassifier,
            "lstm": SimpleLSTMClassifier,
            "cnn-lstm": SimpleCNNLSTMClassifier,
        }.get(SIMPLE_MODEL_TYPE, SimpleCNNClassifier)
        if not os.path.isfile(SIMPLE_MODEL_PATH):
            raise FileNotFoundError(
                f"Simple model not found: {SIMPLE_MODEL_PATH}. "
                "Train with train_simple_models.ipynb and set simple_model.path in train_config.yaml"
            )
        model = load_classifier_checkpoint(model_cls, SIMPLE_MODEL_PATH, DEVICE)
        expected_dim = _get_simple_model_input_dim(model)
        if expected_dim is not None and input_dim != expected_dim:
            raise ValueError(
                f"Input dimension mismatch: preprocessing produced {input_dim} features, "
                f"but simple model expects {expected_dim}. Ensure train_config (normalize, dim_reduction, "
                f"append_prev_char) and data_dir match the setup used to train the simple model."
            )
        model._gik_input_dim = input_dim
        _INFERENCE_MODEL = model
        print(f"Loaded simple model: {SIMPLE_MODEL_TYPE} from {SIMPLE_MODEL_PATH}", file=sys.stderr)
        return model

    model = create_model(
        model_type=TRAINING_CONFIG["model_type"],
        hidden_dim_inner_model=TRAINING_CONFIG["hidden_dim_inner_model"],
        hidden_dim_classification_head=TRAINING_CONFIG["hidden_dim_classification_head"],
        no_layers_classification_head=TRAINING_CONFIG["num_layers"],
        dropout_inner_layers=TRAINING_CONFIG["dropout"],
        inner_model_kwargs=TRAINING_CONFIG["inner_model_prams"],
        output_logits=TRAINING_CONFIG["output_logits"],
        input_dim=input_dim,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()
    model._gik_input_dim = input_dim
    _INFERENCE_MODEL = model
    return model

def run_inference(
    left: np.ndarray = None,
    right: np.ndarray = None,
    left_pointer: int = None,
    right_pointer: int = None,
    prev_char: Any = None,
    history_chars: Optional[list] = None,
):
    """
    prev_char: index of previous char (for preprocess input feature)
    history_chars: list of last (lm_order-1) chars for LM context
    """
    global inference_predictions
    
    append_prev_char = INFERENCE_CONFIG["append_prev_char"]
    if SIMPLE_MODEL_ENABLED:
        append_prev_char = True  # simple models expect prev_char in last 40 dims

    processed_data = preprocess(
        left_data=left,
        right_data=right,
        left_pointer=left_pointer,
        right_pointer=right_pointer,
        prev_char=prev_char,
        mode=EXPERIMENT_MODE,
        max_seq_length=INFERENCE_CONFIG["max_seq_length"],
        normalize=INFERENCE_CONFIG["normalize"],
        apply_filtering=INFERENCE_CONFIG["apply_filtering"],
        apply_dim_reduction=INFERENCE_CONFIG["reduce_dim"],
        dim_red_method=INFERENCE_CONFIG["dim_red_method"],
        dims_ratio=INFERENCE_CONFIG["dims_ratio"],
        root_dir=PROJECT_ROOT,
        training_dataset=PROCESSED_TRAINING_DATA,
        append_prev_char=append_prev_char,
    )

    input_dim = processed_data.shape[2]
    model = _get_inference_model(input_dim)
    x = processed_data if torch.is_tensor(processed_data) else torch.tensor(processed_data, dtype=torch.float32)

    with torch.no_grad():
        if EXPERIMENT_MODE == "classification":
            if SIMPLE_MODEL_ENABLED:
                logits = model(x.to(DEVICE)).squeeze(0).cpu()
            elif TTA_PASSES > 1 or TTA_NOISE_STD > 0.0 or TTA_SCALE_JITTER > 0.0:
                logits = get_logits_single_tta(
                    x, model, str(DEVICE),
                    tta_passes=TTA_PASSES,
                    noise_std=TTA_NOISE_STD,
                    scale_jitter=TTA_SCALE_JITTER,
                )
            else:
                logits = model(x.to(DEVICE)).squeeze(0).cpu()
            lm = _get_inference_lm()
            if lm is not None and LM_INFERENCE_BETA > 0.0:
                history = list(history_chars)[-LM_HISTORY_LEN:] if history_chars else []
                logits = fuse_single_step_logits_with_lm(logits, lm, history, LM_INFERENCE_BETA)
            predicted_idx = logits.argmax(dim=-1).item()
        else:
            predicted_idx = CHAR_TO_INDEX[get_closest_coordinate(model.predict_coords(x.to(DEVICE)), FULL_COORDS)]

    predicted_char = INDEX_TO_CHAR[predicted_idx]
    inference_predictions += predicted_char
    print(predicted_char, end="", flush=True)  # stdout: predictions only
    return predicted_idx
    
async def process_queues(left_queue, right_queue):
    left_win = SlidingWindow()
    right_win = SlidingWindow()

    context_prev = max(0, int(INFERENCE_CONFIG["alignment_prev_windows"]))
    context_future = max(0, int(INFERENCE_CONFIG["alignment_future_windows"]))

    events: List[Dict[str, np.ndarray]] = []
    event_offset = 0
    decoded_abs_idx = 0
    prev_char = None
    history_chars: list = []

    async def decode_ready_events():
        nonlocal event_offset, decoded_abs_idx, prev_char, history_chars
        if not events:
            return
        max_abs_idx = event_offset + len(events) - 1
        while (decoded_abs_idx + context_future) <= max_abs_idx:
            start_abs_idx = max(event_offset, decoded_abs_idx - context_prev)
            end_abs_idx = min(max_abs_idx, decoded_abs_idx + context_future)
            start_i = start_abs_idx - event_offset
            end_i = end_abs_idx - event_offset

            # left_pointer = np.vstack([events[j]["left"] for j in range(start_i, end_i + 1)]).shape[0]
            # right_pointer = np.vstack([events[j]["right"] for j in range(start_i, end_i + 1)]).shape[0]

            # left_ctx = np.vstack([events[j]["left"] for j in range(end_i + 1)])
            # right_ctx = np.vstack([events[j]["right"] for j in range(end_i + 1)])

            left_ctx = np.vstack([events[j]["left"] for j in range(start_i, end_i + 1)])
            right_ctx = np.vstack([events[j]["right"] for j in range(start_i, end_i + 1)])
            prev_char = await asyncio.to_thread(
                run_inference, left_ctx, right_ctx, None, None, prev_char, history_chars
            )
            decoded_abs_idx += 1

            if prev_char is not None and LM_HISTORY_LEN > 0:
                history_chars.append(INDEX_TO_CHAR[prev_char])
                if len(history_chars) > LM_HISTORY_LEN:
                    history_chars.pop(0)

            keep_from_abs_idx = max(event_offset, decoded_abs_idx - context_prev)
            if keep_from_abs_idx > event_offset:
                drop_n = keep_from_abs_idx - event_offset
                events[:] = events[drop_n:]
                event_offset = keep_from_abs_idx
                max_abs_idx = event_offset + len(events) - 1
    left_task = asyncio.create_task(left_queue.get())
    right_task = asyncio.create_task(right_queue.get())
    while True:
        completed, _ = await asyncio.wait(
            [left_task, right_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        hands_with_new_data = []
        for task in completed:
            data, t = task.result()
            data = np.asarray(data, dtype=np.float32)
            data = np.concatenate([data[1:], [t]])

            if task == left_task:
                left_win.append(data)
                left_task = asyncio.create_task(left_queue.get())
                hands_with_new_data.append("left")
            elif task == right_task:
                right_win.append(data)
                right_task = asyncio.create_task(right_queue.get())
                hands_with_new_data.append("right")

        # Check FSR on each hand that received new data; process first trigger only (avoid double-pop)
        for triggered_hand in hands_with_new_data:
            idx = left_win.fsr_detected(fsr_indices=FSR_INDICES) if triggered_hand == "left" else right_win.fsr_detected(fsr_indices=FSR_INDICES)
            if idx is None:
                continue
            chunk = np.stack(left_win.pop_chunk(idx+1)) if triggered_hand == "left" else np.stack(right_win.pop_chunk(idx+1))
            if chunk.shape[0] <= 2:
                continue
            timestamp = chunk[-1][-1]
            opp_win = right_win if triggered_hand == "left" else left_win
            opp_idx = opp_win.timestamp_matched(timestamp=timestamp)
            if opp_idx is not None:
                opp_chunk = np.stack(opp_win.pop_chunk(opp_idx + 1))
            else:
                opp_chunk = np.zeros_like(chunk)
            if triggered_hand == "left":
                events.append({"left": chunk, "right": opp_chunk})
            else:
                events.append({"left": opp_chunk, "right": chunk})
            await decode_ready_events()
            break  # one FSR event per iteration

## ================================================== ##

async def main():
    if SIMPLE_MODEL_ENABLED:
        print(f"Simple model mode: {SIMPLE_MODEL_TYPE} (will load from {SIMPLE_MODEL_PATH} on first inference)", file=sys.stderr)
    print("Waiting for GIK to appear...", file=sys.stderr)

    data_queue_left = asyncio.Queue(MAX_QUEUE_SIZE)
    data_queue_right = asyncio.Queue(MAX_QUEUE_SIZE)

    await asyncio.gather(
        connect(DEVICE_NAME_L, SERVICE_UUID_L, UUID_TX_L, data_queue_left),
        connect(DEVICE_NAME_R, SERVICE_UUID_R, UUID_TX_R, data_queue_right),
        process_queues(data_queue_left, data_queue_right),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
