import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from display import show_frame, close_windows


# Load model
model = YOLO("yolov8n.pt")

# Lấy 1 frame từ video
image = next(
    sv.get_video_frames_generator("vehicles.mp4")
)

# Chạy YOLO
results = model(image)[0]

# Chuyển kết quả sang Detections
detections = sv.Detections.from_ultralytics(results)

print("Số detection ban đầu:", len(detections))
# Chỉ giữ đối tượng confidence > 0.5
detections = detections[detections.confidence > 0.5]

# Chỉ giữ xe hơi (COCO class 2)
detections = detections[detections.class_id == 2]

# Giữ nhiều class: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Kết hợp nhiều điều kiện
detections = detections[
    (detections.confidence > 0.4) & np.isin(detections.class_id, VEHICLE_CLASSES)
]

# Lọc theo diện tích box (loại box quá nhỏ = nhiễu)
detections = detections[detections.area > 1000]

box_annotator = sv.BoxAnnotator(thickness=2)

before = box_annotator.annotate(image.copy(), detections)                 # chưa lọc
after = box_annotator.annotate(image.copy(), detections)                    # đã lọc

cv2.putText(before, f"TRUOC LOC: {len(detections)}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.putText(after, f"SAU LOC: {len(detections)}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

side_by_side = np.hstack([before, after])   # ghép ngang 2 ảnh
show_frame(side_by_side, window_name="Truoc vs Sau khi loc", wait=0)
close_windows()
# Non-Max Suppression thủ công (khi model chưa làm hoặc gộp nhiều model)
detections = detections.with_nms(threshold=0.5, class_agnostic=False)
points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

print ("Tọa độ tâm đáy:")
print(points)