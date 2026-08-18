import cv2
import supervision as sv
from ultralytics import YOLO

from display import show_frame, close_windows

# 1. Load model (yolov8n = nano, nhẹ nhất; tải tự động lần đầu)
model = YOLO("yolov8n.pt")

# 2. Lấy 1 frame từ video làm ảnh thử (hoặc cv2.imread("traffic.jpg"))
image = next(sv.get_video_frames_generator("vehicles.mp4"))

# 3. Chạy inference
results = model(image)[0]

# 4. Chuyển output của YOLO thành sv.Detections
detections = sv.Detections.from_ultralytics(results)

print(detections)

print("Số đối tượng:", len(detections))
print("xyxy (tọa độ box):\n", detections.xyxy)        # ndarray (N, 4)
print("confidence:", detections.confidence)             # ndarray (N,)
print("class_id:", detections.class_id)                 # ndarray (N,)
print("tên class:", detections.data["class_name"])      # ndarray (N,) dạng chuỗi
print("tracker_id:", detections.tracker_id)             # None (chưa track — Bài 5)

# 5. Hiện luôn kết quả lên cửa sổ để đối chiếu với số liệu vừa in
annotated = sv.BoxAnnotator(thickness=2).annotate(image.copy(), detections)

cv2.putText(annotated, f"Phat hien: {len(detections)} doi tuong",
            
            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

show_frame(annotated, wait=0)   # bấm phím bất kỳ để đóng

close_windows()