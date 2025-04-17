import os
import cv2
import requests
from urllib.parse import urlparse
import yt_dlp

# === Cấu hình ===
EXTERNAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external_data"))
VIDEO_DIR = os.path.join(EXTERNAL_DATA_DIR, "videos")
FRAME_DIR = os.path.join(EXTERNAL_DATA_DIR, "frames")
FRAME_INTERVAL_SEC = 2  # Trích mỗi 2 giây

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# === Tải video từ YouTube với yt_dlp, hợp nhất bằng FFmpegMerger ===
def download_youtube_video(url, download_path=VIDEO_DIR):
    create_folder(download_path)
    downloaded_path = None

    def hook(d):
        nonlocal downloaded_path
        if d['status'] == 'finished':
            downloaded_path = d['filename']

    ydl_opts = {
        'format': 'best',
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(download_path, '%(title).100s.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'progress_hooks': [hook],
        # 'postprocessors': [{'key': 'FFmpegMerger'}]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if downloaded_path is None:
            downloaded_path = os.path.join(download_path, f"{info['title']}.{info['ext']}")
        video_title = info.get("title", "video").replace(" ", "_").replace("/", "_")
        print(f"✅ Video đã tải hoặc đã tồn tại: {downloaded_path}")
        return downloaded_path, video_title

# === Tải video từ link trực tiếp (không phải YouTube) ===
def download_generic_video(url, download_path=VIDEO_DIR):
    create_folder(download_path)
    filename = url.split("/")[-1].split("?")[0]
    video_path = os.path.join(download_path, filename)

    if os.path.exists(video_path):
        print(f"⚠️ File đã tồn tại: {video_path}")
        return video_path, os.path.splitext(filename)[0]

    print(f"⬇️ Đang tải video từ: {url}")
    with requests.get(url, stream=True) as r:
        with open(video_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"✅ Đã tải xong: {video_path}")
    return video_path, os.path.splitext(filename)[0]

# === Trích frame từ video mỗi N giây ===
def extract_frames(video_path, output_folder, interval_sec=2):
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        print(f"⚠️ Đã tìm thấy frame tại: {output_folder} → Bỏ qua extract.")
        return

    create_folder(output_folder)
    print(f"🔍 Kiểm tra video tồn tại: {os.path.exists(video_path)} | Đường dẫn: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ OpenCV không thể mở video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("❌ FPS bằng 0, không thể xử lý.")
        return

    interval_frame = int(fps * interval_sec)
    print(f"🎞️ FPS: {fps:.2f} | Mỗi {interval_frame} frame sẽ lưu 1 ảnh.")

    count, saved = 0, 0
    success, frame = cap.read()
    while success:
        if count % interval_frame == 0:
            filename = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1
        success, frame = cap.read()
    cap.release()
    print(f"✅ Đã lưu {saved} frame vào: {output_folder}")

# === Xác định loại input (YouTube hay link trực tiếp) ===
def is_youtube_link(link):
    parsed = urlparse(link)
    return 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc

# === Xử lý 1 input video: tải + extract frame ===
def process_video_input(input_path):
    if is_youtube_link(input_path):
        video_path, video_name = download_youtube_video(input_path)
    elif input_path.startswith("http"):
        video_path, video_name = download_generic_video(input_path)
    else:
        video_path = input_path
        video_name = os.path.splitext(os.path.basename(input_path))[0]

    if not os.path.exists(video_path):
        print(f"❌ Video không tồn tại: {video_path}")
        return

    output_dir = os.path.join(FRAME_DIR, video_name)
    extract_frames(video_path, output_dir)

# === Đọc danh sách link từ file .txt ===
def load_links_from_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# === Chạy chính ===
if __name__ == "__main__":
    test_inputs = [
        "https://www.youtube.com/watch?v=DOKVREgWKbE",
        "https://www.youtube.com/watch?v=EJkn-r-rJJY",
        "https://www.youtube.com/results?search_query=cat+cartoon",
        "https://www.youtube.com/watch?v=_NKcsg8vE_U"
    ]

    for input_link in test_inputs:
        process_video_input(input_link)
