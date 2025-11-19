import cv2
import numpy as np
import random
import shutil
from pathlib import Path

IMG_SIZE = 640
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

def generate_subset(subset_name: str, counts: dict[str, int]):
    img_dir = DATASET_ROOT / subset_name / 'images'
    lbl_dir = DATASET_ROOT / subset_name / 'labels'
    
    for shape_type, count in counts.items():
        for i in range(count):
            bg_color = (255, 255, 255)
            img = np.full((IMG_SIZE, IMG_SIZE, 3), bg_color, dtype=np.uint8)
            x_c, y_c, w, h = draw_shape(img, shape_type)
            class_id = CLASSES[shape_type]
            filename = f"{subset_name}_{shape_type}_{i}"
            cv2.imwrite(str(img_dir / f"{filename}.jpg"), img)
            label_path = lbl_dir / f"{filename}.txt"
            label_path.write_text(f"{class_id} {x_c} {y_c} {w} {h}\n", encoding='utf-8')
            
    print(f"Generated {subset_name} data: {counts}")

def main():
    random.seed(RANDOM_SEED) 
    create_directories()
    generate_subset('train', {'circle': 30, 'square': 30})
    generate_subset('val', {'circle': 3, 'square': 3})
    print(f"\nDataset created successfully in '{DATASET_ROOT}'")
    print(f"Random seed: {RANDOM_SEED}")

if __name__ == "__main__":
    main()