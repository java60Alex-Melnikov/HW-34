from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8m.pt')
    model.train(data='data.yaml', epochs=20, imgsz=640, plots=True)
    print("Training finished. Model saved to runs/detect/train/weights/best.pt")

if __name__ == '__main__':
    train_model()