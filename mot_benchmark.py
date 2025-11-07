import os
import cv2
import glob
import numpy as np
from sort import Sort

# === Path to MOT16 dataset ===
MOT16_PATH = r"C:\Users\shrus\Downloads\MOT16"
SEQUENCE = "MOT16-02"  # change to other sequences if you want (e.g. MOT16-05)

# === Build paths ===
sequence_path = os.path.join(MOT16_PATH, "train", SEQUENCE)
det_file = os.path.join(sequence_path, "det", "det.txt")
frames = sorted(glob.glob(os.path.join(sequence_path, "img1", "*.jpg")))
output_video = os.path.join("output", f"{SEQUENCE}_tracked.avi")

# === Initialize SORT tracker ===
tracker = Sort()

# === Load detections ===
detections = np.loadtxt(det_file, delimiter=',')

# === Initialize video writer ===
first_frame = cv2.imread(frames[0])
height, width = first_frame.shape[:2]
os.makedirs("output", exist_ok=True)
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

# === Process each frame ===
for i, frame_path in enumerate(frames):
    frame = cv2.imread(frame_path)
    frame_num = i + 1

    # Handle float/integer mismatch in frame numbers
    dets_frame = detections[np.round(detections[:, 0]).astype(int) == frame_num]

    # Convert (x, y, w, h) → (x1, y1, x2, y2)
    converted = []
    for d in dets_frame:
        x, y, w, h, score = d[2], d[3], d[4], d[5], d[6]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        converted.append([x1, y1, x2, y2, score])

    dets = np.array(converted)

    # Update SORT tracker
    if len(dets) > 0:
        trackers = tracker.update(dets)
    else:
        trackers = tracker.update()

    # Draw boxes and IDs
    for d in trackers:
        x1, y1, x2, y2, track_id = [int(v) for v in d]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(track_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    print(f"Processed frame {frame_num}/{len(frames)}")

out.release()
cv2.destroyAllWindows()
print(f"✅ Tracking complete. Output saved at: {output_video}")
