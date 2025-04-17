import os
import cv2
import random
from shutil import copyfile

# === Định nghĩa thư mục dữ liệu chung ===
EXTERNAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external_data"))
ENSEMBLE_LABEL_DIR = os.path.join(EXTERNAL_DATA_DIR, "ensemble_output")
FRAME_DIR = os.path.join(EXTERNAL_DATA_DIR, "frames")
OUT_DIR = os.path.join(EXTERNAL_DATA_DIR, "custom_dataset_yolo")
TRAIN_RATIO = 0.8  # 80% train, 20% val

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_to_yolo_format(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return [x_center, y_center, width, height]

def prepare_dataset():
    image_label_pairs = []

    for video_name in os.listdir(ENSEMBLE_LABEL_DIR):
        label_dir = os.path.join(ENSEMBLE_LABEL_DIR, video_name)
        image_dir = os.path.join(FRAME_DIR, video_name)

        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue

            label_path = os.path.join(label_dir, fname)
            image_path = os.path.join(image_dir, fname.replace(".txt", ".jpg"))

            if os.path.exists(image_path):
                image_label_pairs.append((image_path, label_path))

    print(f"🧩 Tổng số ảnh hợp lệ: {len(image_label_pairs)}")

    random.shuffle(image_label_pairs)
    split_idx = int(len(image_label_pairs) * TRAIN_RATIO)
    train_set = image_label_pairs[:split_idx]
    val_set = image_label_pairs[split_idx:]

    for split_name, data in [("train", train_set), ("val", val_set)]:
        img_out = os.path.join(OUT_DIR, "images", split_name)
        lbl_out = os.path.join(OUT_DIR, "labels", split_name)
        create_folder(img_out)
        create_folder(lbl_out)

        for img_path, lbl_path in data:
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            fname = os.path.basename(img_path)
            new_img_path = os.path.join(img_out, fname)
            new_lbl_path = os.path.join(lbl_out, fname.replace(".jpg", ".txt"))
            copyfile(img_path, new_img_path)

            with open(lbl_path, 'r') as f_in, open(new_lbl_path, 'w') as f_out:
                for line in f_in:
                    cls, conf, x1, y1, x2, y2 = line.strip().split()
                    cls = int(cls)
                    bbox = list(map(float, [x1, y1, x2, y2]))
                    norm_bbox = convert_to_yolo_format(bbox, w, h)
                    norm_str = " ".join(f"{v:.6f}" for v in norm_bbox)
                    f_out.write(f"{cls} {norm_str}\n")

    print(f"✅ Dataset đã tạo tại: {OUT_DIR}/images & /labels")

def write_dataset_yaml():
    yaml_path = os.path.join(OUT_DIR, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(
            f"""train: {OUT_DIR}/images/train
val: {OUT_DIR}/images/val

nc: 5
names: ['Person', 'Dog', 'Cat', 'Car', 'Tree']
"""
        )
    print(f"📄 Đã tạo file dataset.yaml")

if __name__ == "__main__":
    prepare_dataset()
    write_dataset_yaml()
