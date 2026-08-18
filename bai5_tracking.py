import numpy as np
import supervision as sv
from ultralytics import YOLO

from display import show_frame, close_windows

SOURCE_VIDEO = "vehicles.mp4"
TARGET_VIDEO = "bai5_tracking.mp4"

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
smoother = sv.DetectionsSmoother(length=5)

box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)

trace_annotator = sv.TraceAnnotator(
    trace_length=video_info.fps * 2,   # lưu vệt 2 giây
    position=sv.Position.BOTTOM_CENTER,
)

#Xử lý từng frame
def process_frame(frame):
    #yolo detection
    results = model(frame, verbose=False)[0]
    #Chuyển sang detections
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]

    # Cập nhật ByteTrack
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)
    # Tạo label gồm ID + tên class
    labels = [
    f"#{tid} {model.names[class_id]}"
    for tid, class_id in zip(
        detections.tracker_id,
        detections.class_id
    )
]
    #Vẽ box
    annotated = frame.copy()
    annotated = box_annotator.annotate(annotated, detections)
    #Vẽ label
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    return annotated
#Đọc video, xử lý, vừa xem vừa ghi
with sv.VideoSink(target_path=TARGET_VIDEO, video_info=video_info) as sink:
    for frame in sv.get_video_frames_generator(SOURCE_VIDEO):
         annotated = process_frame(frame)
         sink.write_frame(annotated)              # ghi file
         if not show_frame(annotated):            # hiện cửa sổ; bấm Q -> dừng
            print("Nguoi dung bam Q - dung som.")
            break
    
close_windows()
print("Xong! Video da luu tai:", TARGET_VIDEO)