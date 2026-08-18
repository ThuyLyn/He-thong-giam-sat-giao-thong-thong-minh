import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

from display import show_frame, close_windows

SOURCE_VIDEO = "vehicles.mp4"
TARGET_VIDEO = "bai6_linezone.mp4"

VEHICLE_CLASSES = [2, 3, 5, 7]

model = YOLO("yolov8n.pt")

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)
#Tạo ByteTrack
tracker = sv.ByteTrack(
    frame_rate=video_info.fps,
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
)
#Làm mượt kết quả tracking
smoother = sv.DetectionsSmoother(length=5)
# Vạch ngang giữa khung hình (chỉnh tọa độ theo video của em)
START = sv.Point(0, video_info.height // 2)
END = sv.Point(video_info.width, video_info.height // 2)
#Tạo Linezone
line_zone = sv.LineZone(
    start=START,
    end=END,
    triggering_anchors=[sv.Position.BOTTOM_CENTER],  # xét điểm chạm đất
    minimum_crossing_threshold=2,   # cần 2 frame xác nhận — chống đếm trùng do jitter
)

line_annotator = sv.LineZoneAnnotator(
    thickness=2,
    text_scale=0.8,
    custom_in_text="Vao",
    custom_out_text="Ra",
)

#Annotator
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)

trace_annotator = sv.TraceAnnotator(
    trace_length=video_info.fps * 2,  
    position=sv.Position.BOTTOM_CENTER,
)
counts = defaultdict(int)

def update_counts(detections, line_zone):
    crossed_in, crossed_out = line_zone.trigger(detections)
    for name in detections.data["class_name"][crossed_in]:
        counts[f"{name}_in"] += 1
    for name in detections.data["class_name"][crossed_out]:
        counts[f"{name}_out"] += 1

def draw_stats(frame, stats: dict, origin=(20, 50)):
    """Vẽ bảng thống kê nền đen mờ ở góc trên trái — nhìn rõ trên mọi video."""
    x, y = origin
    line_h = 45
    h = line_h * (len(stats) + 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 40), (x + 420, y - 40 + h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)   # nền đen mờ 60%
    for i, (k, v) in enumerate(stats.items()):
        cv2.putText(frame, f"{k}: {v}", (x, y + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return frame
def process_frame(frame):
    #yolo detection
    results = model(frame, verbose=False)[0]
    #Chuyển sang detections
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]

    # Cập nhật ByteTrack
    detections = tracker.update_with_detections(detections)
    #Làm mượt kết quả tracking
    detections = smoother.update_with_detections(detections)
    #Tạo class name
    detections.data["class_name"] = np.array([
        model.names[int(class_id)]
        for class_id in detections.class_id
    ])
    #Trigger được gọi duy nhất một lần
    update_counts(detections, line_zone)
    #Tạo label ID và CLASS
    labels = [
        f"#{tid} {model.names[class_id]}"
        for tid, class_id in zip(
            detections.tracker_id,
            detections.class_id
        )
    ]
    #Frame kết quả
    annotated = frame.copy()
    #Vẽ bounding box
    annotated = box_annotator.annotate(annotated, detections)
    #Vẽ label
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    #Vẽ trace
    annotated = trace_annotator.annotate(annotated, detections)
    #Vẽ vạch vào / ra
    annotated = line_annotator.annotate(annotated, line_counter=line_zone)
    #Vẽ bảng thống kê
    annotated = draw_stats(annotated, counts)
    return annotated

#Đọc video, xử lý
with sv.VideoSink(target_path=TARGET_VIDEO, video_info=video_info) as sink:
    for frame in sv.get_video_frames_generator(SOURCE_VIDEO):
         annotated = process_frame(frame)
         sink.write_frame(annotated)              # ghi file
         if not show_frame(annotated):            # hiện cửa sổ; bấm Q -> dừng
            print("Nguoi dung bam Q - dung som.")
            break
    
close_windows()
print("Xong! Video da luu tai:", TARGET_VIDEO)

