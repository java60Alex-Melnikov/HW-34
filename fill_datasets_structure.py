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

def draw_shape(img: np.ndarray, shape_type: str) -> tuple[float, float, float, float]:
    h, w = img.shape[:2]
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    size = random.randint(50, 150)  
    x = random.randint(size // 2, w - size // 2)
    y = random.randint(size // 2, h - size // 2)
    
    if shape_type == 'circle':
        radius = size // 2
        cv2.circle(img, (x, y), radius, color, -1)
        bbox_w, bbox_h = size, size
    else:
        top_left_x = x - size // 2
        top_left_y = y - size // 2
        cv2.rectangle(img, (top_left_x, top_left_y), (top_left_x + size, top_left_y + size), color, -1)
        bbox_w, bbox_h = size, size

    return x / w, y / h, bbox_w / w, bbox_h / h