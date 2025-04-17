import os
from ultralytics import YOLO

# === Cấu hình ===
EXTERNAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external_data"))
YOLO_BACKBONE = "yolov5s.pt"  # Hoặc yolov8n.pt, yolov5m.pt, ...
DATA_YAML = os.path.join(EXTERNAL_DATA_DIR, "custom_dataset_yolo/dataset.yaml")
EPOCHS = 50
IMG_SIZE = 640
PROJECT_DIR = os.path.join(EXTERNAL_DATA_DIR, "training_output")
RUN_NAME = "custom_animodel"

# === Nhãn class của bạn
CLASS_NAMES = ['Person', 'Dog', 'Cat', 'Car', 'Tree']

def train_custom_yolo():
    print("🚀 Bắt đầu khởi tạo và sửa mô hình YOLO...")

    model = YOLO(YOLO_BACKBONE)
    model.model.nc = len(CLASS_NAMES)
    model.model.names = CLASS_NAMES

    print(f"🛠 Model đã được chỉnh sửa: {len(CLASS_NAMES)} lớp → {CLASS_NAMES}")

    print("🚀 Bắt đầu huấn luyện...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True
    )

if __name__ == "__main__":
    train_custom_yolo()
