import os
import random
import shutil

SOURCE_FRAMES = "frames"
DESTINATION = "golden_set"
SAMPLES_PER_VIDEO = 20  # ✅ Bạn có thể đổi số này tùy theo nhu cầu

def sample_frames():
    os.makedirs(DESTINATION, exist_ok=True)
    for video_name in os.listdir(SOURCE_FRAMES):
        video_path = os.path.join(SOURCE_FRAMES, video_name)
        if not os.path.isdir(video_path):
            continue
        all_frames = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])
        if not all_frames:
            continue
        sampled = random.sample(all_frames, min(SAMPLES_PER_VIDEO, len(all_frames)))
        out_dir = os.path.join(DESTINATION, video_name)
        os.makedirs(out_dir, exist_ok=True)
        for fname in sampled:
            src = os.path.join(video_path, fname)
            dst = os.path.join(out_dir, fname)
            shutil.copyfile(src, dst)
    print(f"✅ Đã chọn {SAMPLES_PER_VIDEO} ảnh mỗi video vào thư mục `{DESTINATION}/`")

if __name__ == "__main__":
    sample_frames()
