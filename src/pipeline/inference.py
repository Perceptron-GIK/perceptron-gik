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

async def inference_task(queue: asyncio.Queue,
                         model,
                         window_size: int,
                         feature_dim: int,
                         device="cpu"):

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