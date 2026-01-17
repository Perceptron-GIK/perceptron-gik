import numpy as np

def normalise(x):
    try:
        return x / np.linalg.norm(x)
    except:
        return x