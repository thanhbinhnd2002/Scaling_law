import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# === Định nghĩa thư mục dữ liệu chung ===
EXTERNAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external_data"))

class MultiModelDetector:
    def __init__(self, model_paths):
        self.models = [YOLO(p) for p in model_paths]
        self.model_names = [os.path.basename(p).split('.')[0] for p in model_paths]

        # Mapping từ COCO ID → Custom ID
        self.class_map = {
            0: 0,    # person
            16: 1,   # dog
            15: 2,   # cat
            2: 3,    # car
            18: 4    # tree
        }

    def detect_image(self, image_path):
        all_results = []
        for model, name in zip(self.models, self.model_names):
            results = model(image_path)[0]
            detections = []
            for box in results.boxes:
                orig_cls = int(box.cls.item())
                if orig_cls not in self.class_map:
                    continue  # Bỏ qua object không thuộc class mong muốn

                custom_cls = self.class_map[orig_cls]
                conf = float(box.conf.item())
                xyxy = [round(x, 2) for x in box.xyxy.tolist()[0]]  # [x1, y1, x2, y2]

                detections.append({
                    "model": name,
                    "class": custom_cls,
                    "conf": conf,
                    "bbox": xyxy
                })
            all_results.append({
                "model": name,
                "detections": detections
            })
        return all_results

    def detect_folder(self, frame_root=os.path.join(EXTERNAL_DATA_DIR, "frames"), output_root=os.path.join(EXTERNAL_DATA_DIR, "raw_detections")):
        for video_name in os.listdir(frame_root):
            input_dir = os.path.join(frame_root, video_name)
            if not os.path.isdir(input_dir):
                continue

            print(f"\n🚀 Đang xử lý video: {video_name}")
            image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

            for filename in tqdm(image_files, desc=f"🔍 {video_name}", unit="img"):
                img_path = os.path.join(input_dir, filename)
                all_model_results = self.detect_image(img_path)

                for result in all_model_results:
                    model_name = result["model"]
                    output_dir = os.path.join(output_root, model_name, video_name)
                    os.makedirs(output_dir, exist_ok=True)

                    label_txt = filename.replace(".jpg", ".txt")
                    label_path = os.path.join(output_dir, label_txt)

                    with open(label_path, 'w') as f:
                        for det in result["detections"]:
                            cls = det["class"]
                            conf = det["conf"]
                            x1, y1, x2, y2 = det["bbox"]
                            f.write(f"{cls} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

if __name__ == "__main__":
    model_paths = [
        "yolov5s.pt",
        "yolov8n.pt"
    ]

    detector = MultiModelDetector(model_paths)
    detector.detect_folder()
