import os
import cv2

# === Gán màu cho từng class ID ===
COLORS = {
    0: (0, 255, 0),     # Person - xanh lá
    1: (0, 0, 255),     # Dog - đỏ
    2: (255, 0, 0),     # Cat - xanh dương
    3: (255, 255, 0),   # Car - vàng
    4: (0, 255, 255)    # Tree - xanh biển
}

CLASS_NAMES = {
    0: "Person",
    1: "Dog",
    2: "Cat",
    3: "Car",
    4: "Tree"
}

# === Định nghĩa thư mục dữ liệu chung ===
EXTERNAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external_data"))


def draw_boxes_on_image(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return None

    if not os.path.exists(label_path):
        return image

    with open(label_path, 'r') as f:
        for line in f:
            cls, conf, x1, y1, x2, y2 = line.strip().split()
            cls = int(cls)
            conf = float(conf)
            x1, y1, x2, y2 = map(int, map(float, [x1, y1, x2, y2]))

            color = COLORS.get(cls, (200, 200, 200))
            label = f"{CLASS_NAMES.get(cls, 'Unknown')} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


def process_video(video_name, frame_root=os.path.join(EXTERNAL_DATA_DIR, "frames"), label_root=os.path.join(EXTERNAL_DATA_DIR, "ensemble_output"), output_root=os.path.join(EXTERNAL_DATA_DIR, "ensemble_output_images")):
    input_dir = os.path.join(frame_root, video_name)
    label_dir = os.path.join(label_root, video_name)
    output_dir = os.path.join(output_root, video_name)

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

    for fname in image_files:
        image_path = os.path.join(input_dir, fname)
        label_path = os.path.join(label_dir, fname.replace(".jpg", ".txt"))
        result = draw_boxes_on_image(image_path, label_path)

        if result is not None:
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, result)

    print(f"✅ Đã xuất ảnh có bbox: {output_dir}")


if __name__ == "__main__":
    frame_root = os.path.join(EXTERNAL_DATA_DIR, "frames")
    label_root = os.path.join(EXTERNAL_DATA_DIR, "ensemble_output")
    output_root = os.path.join(EXTERNAL_DATA_DIR, "ensemble_output_images")

    video_list = os.listdir(label_root)

    for video in video_list:
        process_video(video, frame_root, label_root, output_root)
