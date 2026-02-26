import asyncio
import numpy as np
import torch
from .sliding_window import SlidingWindow

def load_model(model_class, model_path: str, device="cpu"):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

async def combine_queues(
        left_queue: asyncio.Queue,
        right_queue: asyncio.Queue,
        combined_queue: asyncio.Queue,
):
    while True:
        data_l, t_l = await left_queue.get()
        data_l = np.asarray(data_l, dtype=np.float32)
        data_l = np.concatenate([data_l, [t_l]])

        data_r, t_r = await right_queue.get()
        data_r = np.asarray(data_r, dtype=np.float32)
        data_r = np.concatenate([data_r, [t_r]])

        # Check if there is any non-zero FSR data.
        # If yes, align and preprocess. If not, keep waiting.

        # CALL ALIGNMENT HERE

        combined = np.concatenate([data_l, data_r], axis=-1)

        # Do preprocessing here
        combined = preprocess(combined)

        await combined_queue.put((combined, min(t_l, t_r)))
)

async def inference_task(
        queue: asyncio.Queue,
        model,
        window_size: int,
        feature_dim: int,
        device="cpu"
):

    window = SlidingWindow(window_size, feature_dim)

    while True:
        data, t = await queue.get()

        # TO-DO: Add preprocessing and dimensionality reduction
        window.add(data)

        if window.is_ready():
            input_tensor = window.get_tensor().to(device)

            with torch.no_grad():
                output = model(input_tensor)

            print(f"Prediction: {output}")