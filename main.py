import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import clip
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Path to the cropped image you provided
cropped_ref_img_path = "/content/Traget Person.png"
if not os.path.exists(cropped_ref_img_path):
    print(f"Error: The cropped reference image was not found at {cropped_ref_img_path}")
    exit()

pil_cropped_ref = Image.open(cropped_ref_img_path).convert("RGB")
ref_img_input = preprocess(pil_cropped_ref).unsqueeze(0).to(device)

with torch.no_grad():
    ref_features = clip_model.encode_image(ref_img_input)
    ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)

yolo_model = YOLO("yolov8n.pt")
yolo_model.to(device)

print("✅ Reference features successfully extracted from the cropped image.")

# --- STEP 2: Video Tracking with a robust, ID-based approach ---
tracker = DeepSort(max_age=60, n_init=5)
video_path = "/content/Task_Video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("tracked_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

flow_line = []
target_track_id = None
frame_idx = 0
first_appearance = None
last_known_position = None

# A confidence threshold to lock onto a person
INITIAL_LOCK_THRESHOLD = 0.65

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(frame, conf=0.4, classes=0)
    detections = results[0].boxes
    person_detections = []

    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        person_detections.append(([x1, y1, x2, y2], conf))

    tracks = tracker.update_tracks(person_detections, frame=frame)

    if target_track_id is None:
        best_match_id = -1
        max_frame_similarity = -1

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            person_crop = frame[t:b, l:r]

            if person_crop.size == 0:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            image_input = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = torch.nn.functional.cosine_similarity(image_features, ref_features).item()

            if similarity > max_frame_similarity:
                max_frame_similarity = similarity
                best_match_id = track_id

        if max_frame_similarity > INITIAL_LOCK_THRESHOLD:
            target_track_id = best_match_id
            first_appearance = frame_idx / fps
            print(f"✅ Locked onto target person with ID {target_track_id} with similarity {max_frame_similarity:.2f}.")

    # --- Only draw bounding box and motion line for the tracked person ---
    for track in tracks:
        if track.is_confirmed() and track.track_id == target_track_id:
            l, t, r, b = map(int, track.to_ltrb())

            color = (0, 255, 0) # Green for target
            center = ((l + r) // 2, (t + b) // 2)
            flow_line.append(center)
            last_known_position = center

            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f"ID: {track.track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Handle occlusion by continuing to draw from the last known position
    if target_track_id is not None and not any(track.track_id == target_track_id for track in tracks) and last_known_position:
        flow_line.append(last_known_position)

    for i in range(1, len(flow_line)):
        cv2.line(frame, flow_line[i - 1], flow_line[i], (255, 0, 0), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

if first_appearance is not None:
    print(f"✅ Final tracked person with ID {target_track_id} first appears at {first_appearance:.2f} seconds.")
else:
    print("❌ Target person not found in video.")