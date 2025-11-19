import cv2
import numpy as np
import random
import shutil
from pathlib import Path

IMG_SIZE = 256
CLASSES = {'circle': 0, 'square': 1}
DATASET_ROOT = Path('datasets')
RANDOM_SEED = random.randint(0, 10000)

def create_directories():
    for subset in ['train', 'val']:
        for kind in ['images', 'labels']:
            path = DATASET_ROOT / subset / kind
            if path.exists():
                shutil.rmtree(path) 
            path.mkdir(parents=True, exist_ok=True)