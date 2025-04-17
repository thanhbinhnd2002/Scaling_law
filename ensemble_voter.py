import os
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# === Định nghĩa thư mục dữ liệu chung ===
EXTERNAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external_data"))

# === Cấu hình ===
IOU_THRESHOLD = 0.5
ALLOWED_CLASSES = {0, 1, 2, 3, 4}


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)


def load_detections(file_path):
    detections = []
    if not os.path.exists(file_path):
        return detections
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            cls, conf, x1, y1, x2, y2 = parts
            cls = int(cls)
            if cls not in ALLOWED_CLASSES:
                continue
            detections.append({
                "class": cls,
                "conf": float(conf),
                "bbox": list(map(float, [x1, y1, x2, y2]))
            })
    return detections


def ensemble_detections(detections_per_model):
    merged = []
    used = set()

    for i, det_list in enumerate(detections_per_model):
        for d in det_list:
            if id(d) in used:
                continue
            group = [d]
            used.add(id(d))
            for j in range(i + 1, len(detections_per_model)):
                for d2 in detections_per_model[j]:
                    if id(d2) in used:
                        continue
                    if iou(d["bbox"], d2["bbox"]) >= IOU_THRESHOLD:
                        group.append(d2)
                        used.add(id(d2))

            labels = [g["class"] for g in group]
            final_class = Counter(labels).most_common(1)[0][0]

            weights = [g["conf"] for g in group]
            bboxes = np.array([g["bbox"] for g in group])
            weighted_bbox = np.average(bboxes, axis=0, weights=weights)

            merged.append({
                "class": final_class,
                "conf": round(np.mean(weights), 4),
                "bbox": [round(x, 2) for x in weighted_bbox.tolist()]
            })
    return merged


def process_video(video_name, model_dirs, input_root=os.path.join(EXTERNAL_DATA_DIR, "raw_detections"), output_root=os.path.join(EXTERNAL_DATA_DIR, "ensemble_output")):
    input_files = defaultdict(list)
    for model in model_dirs:
        path = os.path.join(input_root, model, video_name)
        if not os.path.isdir(path):
            continue
        for fname in os.listdir(path):
            if fname.endswith(".txt"):
                input_files[fname].append(os.path.join(path, fname))

    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(sorted(input_files), desc=f"🧠 Ensemble {video_name}", unit="img"):
        detections_per_model = []
        for file_path in input_files[fname]:
            detections = load_detections(file_path)
            detections_per_model.append(detections)

        merged = ensemble_detections(detections_per_model)

        out_path = os.path.join(output_dir, fname)
        with open(out_path, 'w') as f:
            for det in merged:
                cls = det["class"]
                conf = det["conf"]
                x1, y1, x2, y2 = det["bbox"]
                f.write(f"{cls} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")


if __name__ == "__main__":
    model_names = ["yolov5s", "yolov8n"]
    video_names = os.listdir(os.path.join(EXTERNAL_DATA_DIR, "raw_detections", model_names[0]))

    for video in video_names:
        process_video(video, model_names)
