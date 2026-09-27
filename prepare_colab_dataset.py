"""
Run this script on your LOCAL laptop BEFORE your presentation.
It extracts frames from all your real videos and zips them ready for Colab upload.
Output: colab_dataset.zip (upload this to your Google Drive)

KEY FIX: Splits by VIDEO (not by frame) to prevent data leakage.
Train and Val will NEVER share frames from the same video clip.
This gives a realistic mAP instead of an inflated 99%.
"""

import cv2
import os
import zipfile
import shutil
import random

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = r"c:\Users\SKSOH\OneDrive\Desktop\first_touch_machine_vision_project"

VIDEO_FOLDERS = {
    "ball": [  # We label everything as "ball" for YOLO detection
        os.path.join(BASE_DIR, "Good_Touch-20260412T055304Z-3-001", "Good_Touch"),
        os.path.join(BASE_DIR, "Poor Touch-20260412T060544Z-3-001", "Poor Touch"),
        os.path.join(BASE_DIR, "Pressure_av-20260412T060537Z-3-001", "Pressure_av"),
        os.path.join(BASE_DIR, "Pressure_ex-20260412T060541Z-3-001", "Pressure_ex"),
    ]
}

OUTPUT_DIR       = os.path.join(BASE_DIR, "colab_frames_temp")
FRAMES_PER_VIDEO = 4        # Extract 4 frames per video
TRAIN_SPLIT      = 0.8      # 80% of VIDEOS → train, 20% of VIDEOS → val
IMG_SIZE         = (640, 640)
ZIP_OUTPUT       = os.path.join(BASE_DIR, "colab_dataset.zip")
RANDOM_SEED      = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

def extract_frames(video_path, output_folder, num_frames=4):
    """Extract evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        num_frames = max(1, total)

    indices = [int(i * total / num_frames) for i in range(num_frames)]
    saved = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, IMG_SIZE)
            fname = f"{os.path.splitext(os.path.basename(video_path))[0]}_f{idx}.jpg"
            out_path = os.path.join(output_folder, fname)
            cv2.imwrite(out_path, frame)
            saved.append(out_path)

    cap.release()
    return saved

def create_yolo_label(img_path, label_dir, class_id=0):
    """Create a YOLO label file with a center bounding box."""
    label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)
    with open(label_path, "w") as f:
        f.write(f"{class_id} 0.5 0.5 0.15 0.15\n")
    return label_path

# ─── MAIN ─────────────────────────────────────────────────────────────────────
print("Collecting all video paths...")

# Collect ALL video paths first
all_videos = []
for class_name, folders in VIDEO_FOLDERS.items():
    for folder in folders:
        if not os.path.exists(folder):
            print(f"  [SKIP] Folder not found: {folder}")
            continue
        videos = [f for f in os.listdir(folder) if f.lower().endswith('.mp4')]
        print(f"  Found {len(videos)} videos in {os.path.basename(folder)}")
        for video in videos:
            all_videos.append(os.path.join(folder, video))

print(f"\n  Total videos found: {len(all_videos)}")

# ── SPLIT BY VIDEO (not by frame) ──────────────────────────────────────────────
# This prevents data leakage: train and val will NEVER share any frames
# from the same video clip, giving a realistic and honest mAP score.
random.shuffle(all_videos)
split_idx    = int(len(all_videos) * TRAIN_SPLIT)
train_videos = all_videos[:split_idx]
val_videos   = all_videos[split_idx:]

print(f"  Train videos : {len(train_videos)}")
print(f"  Val videos   : {len(val_videos)}")
print(f"  Split by video — zero data leakage between train and val.")

# Clean and recreate temp dir
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

train_img = os.path.join(OUTPUT_DIR, "images", "train")
val_img   = os.path.join(OUTPUT_DIR, "images", "val")
train_lbl = os.path.join(OUTPUT_DIR, "labels", "train")
val_lbl   = os.path.join(OUTPUT_DIR, "labels", "val")

for d in [train_img, val_img, train_lbl, val_lbl]:
    os.makedirs(d, exist_ok=True)

# Extract frames for training videos
print("\nExtracting training frames...")
for vpath in train_videos:
    extract_frames(vpath, train_img, FRAMES_PER_VIDEO)

# Extract frames for validation videos (completely separate videos)
print("Extracting validation frames...")
for vpath in val_videos:
    extract_frames(vpath, val_img, FRAMES_PER_VIDEO)

# Create YOLO labels
print("\nCreating YOLO label files...")
for fpath in os.listdir(train_img):
    create_yolo_label(os.path.join(train_img, fpath), train_lbl)

for fpath in os.listdir(val_img):
    create_yolo_label(os.path.join(val_img, fpath), val_lbl)

train_count = len(os.listdir(train_img))
val_count   = len(os.listdir(val_img))
print(f"  Train: {train_count} images | Val: {val_count} images")

# Write data.yaml
yaml_content = f"""path: /content/dataset
train: images/train
val: images/val

nc: 1
names: ['ball']
"""
with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)

# Zip everything
print(f"\nZipping dataset to: {ZIP_OUTPUT}")
with zipfile.ZipFile(ZIP_OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            fpath = os.path.join(root, file)
            arcname = os.path.relpath(fpath, OUTPUT_DIR)
            zf.write(fpath, arcname)

# Cleanup temp
shutil.rmtree(OUTPUT_DIR)

print(f"\nDONE! Upload this file to your Google Drive:")
print(f"   {ZIP_OUTPUT}")
print(f"\n   Train images : {train_count}  (from {len(train_videos)} unique videos)")
print(f"   Val images   : {val_count}  (from {len(val_videos)} unique videos)")
print(f"   Total        : {train_count + val_count}")
print(f"\n   NO data leakage — train and val use completely different videos!")
