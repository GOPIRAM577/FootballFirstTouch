import os
import cv2
import random

def extract_frames_for_labeling(input_dir="data/inputs", output_dir="data/dataset_frames", frames_per_video=8):
    """
    Extracts a few random frames from every video in the input directory
    so you can easily upload them to Roboflow for labeling.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.mp4'):
                video_files.append(os.path.join(root, f))
                
    if not video_files:
        print(f"No videos found in {input_dir}")
        return
        
    print(f"Found {len(video_files)} videos. Extracting {frames_per_video} frames from each...")
    
    total_extracted = 0
    
    for vid_path in video_files:
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            continue
            
        # Select random frame indices to extract
        # We pick random frames so we get a good variety of ball positions
        if total_frames <= frames_per_video:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = sorted(random.sample(range(total_frames), frames_per_video))
            
        base_name = os.path.splitext(os.path.basename(vid_path))[0].replace(" ", "_").replace(":", "-")
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                out_path = os.path.join(output_dir, f"{base_name}_frame_{idx}.jpg")
                cv2.imwrite(out_path, frame)
                total_extracted += 1
                
        cap.release()
        print(f"  Extracted frames from {base_name}")
        
    print(f"\n✅ Done! {total_extracted} total images saved to '{output_dir}'.")
    print("You can now upload the 'dataset_frames' folder to Roboflow!")

if __name__ == "__main__":
    extract_frames_for_labeling()
