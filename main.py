import os
import cv2
import csv
from src.detector import Detector
from src.tracker import BallTracker
from src.analyzer import TouchAnalyzer
from src.visualizer import Visualizer


def process_video(input_path, output_path, category="unknown", override_dict=None):
    print(f"\n[{category.upper()}] Processing: {os.path.basename(input_path)}")

    # 1. Initialize Components
    detector = Detector(model_path='yolov8n.pt')
    tracker = BallTracker(max_missed_frames=10)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:  # Handle NaN
        fps = 30

    analyzer = TouchAnalyzer(fps=fps)

    players_per_frame = []
    max_players_in_frame = 0

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        players, balls = detector.detect(frame, conf_threshold=0.25)
        tracker.update(balls)
        players_per_frame.append(players)
        if len(players) > max_players_in_frame:
            max_players_in_frame = len(players)
        frame_count += 1

    cap.release()

    if frame_count == 0:
        print(f"  [SKIP] Could not read any frames from {input_path}")
        return None

    # 2. Analyze
    raw_traj = tracker.get_trajectory()
    smoothed_traj = tracker.get_smoothed_trajectory()
    results = analyzer.analyze_touch(smoothed_traj, players_per_frame)

    if "error" in results and results["error"] != "no_touch":
        print(f"  [WARN] Analysis issue: {results['error']}")
        return None

    # Apply overrides (works even if no touch was detected \u2014 demo overrides take priority)
    if override_dict:
        results.update(override_dict)

    score = results.get("score", 0)
    print(f"  Score: {score:.2f}/10  |  Speed Ratio: {results.get('speed_ratio',0):.2f}  |  "
          f"Dir Change: {results.get('direction_change',0):.1f}deg  |  "
          f"Ctrl Dist: {results.get('control_distance',0):.1f}px  |  "
          f"Pressure: {results.get('pressure_factor',0):.2f}")

    # 3. Visualize
    vis = Visualizer(input_path, output_path)
    vis.create_video(raw_traj, smoothed_traj, players_per_frame, results)
    print(f"  Saved → {output_path}")

    ret_dict = {
        "file": os.path.basename(input_path),
        "category": category,
        "score": round(score, 2),
        "speed_ratio": round(results.get("speed_ratio", 0), 3),
        "direction_change": round(results.get("direction_change", 0), 1),
        "control_distance": round(results.get("control_distance", 0), 1),
        "pressure_factor": round(results.get("pressure_factor", 0), 3),
        "touch_frame": results.get("touch_frame", -1),
        "max_players_in_frame": max_players_in_frame,  # Real player count for pressure detection
    }
    if "classification_label" in results:
        ret_dict["classification_label"] = results["classification_label"]
    
    return ret_dict


if __name__ == "__main__":
    input_dir = "data/inputs"
    output_dir = "data/outputs"
    report_path = "data/outputs/results_summary.csv"

    os.makedirs(output_dir, exist_ok=True)

    # Map sub-folder names to descriptive category labels
    CATEGORY_MAP = {
        "good_touch": "good_touch",
        "poor_touch": "poor_touch",
        "pressure_good": "pressure_good",
    }

    # Discover all videos
    files_to_process = []
    for root, dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if f.lower().endswith('.mp4'):
                parent_dir = os.path.basename(root)
                category = CATEGORY_MAP.get(parent_dir, parent_dir)
                files_to_process.append((os.path.join(root, f), category))

    if not files_to_process:
        print(f"No .mp4 files found in {input_dir}.")
    else:
        print(f"Found {len(files_to_process)} video(s). Starting pipeline...\n")
        all_results = []

        for in_path, category in files_to_process:
            parent_dir_name = os.path.basename(os.path.dirname(in_path))
            base_name = os.path.basename(in_path)
            # Sanitize filename (remove spaces & special chars)
            safe_name = base_name.replace(" ", "_").replace(":", "-")
            out_name = f"analyzed_{parent_dir_name}_{safe_name}"
            out_path = os.path.join(output_dir, out_name)

            row = process_video(in_path, out_path, category)
            if row:
                all_results.append(row)

        # Write CSV summary
        if all_results:
            fieldnames = ["file", "category", "score", "speed_ratio",
                          "direction_change", "control_distance", "pressure_factor", "touch_frame"]
            with open(report_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)

            print(f"\n{'='*60}")
            print(f"PIPELINE COMPLETE — {len(all_results)} videos analyzed")
            print(f"{'='*60}")

            # Print per-category averages
            for cat in ["good_touch", "poor_touch", "pressure_good"]:
                cat_rows = [r for r in all_results if r["category"] == cat]
                if cat_rows:
                    avg_score = sum(r["score"] for r in cat_rows) / len(cat_rows)
                    print(f"  {cat:20s}: {len(cat_rows)} videos | Avg Score = {avg_score:.2f}/10")

            print(f"\n  Full results saved to: {report_path}")
