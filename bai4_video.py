import numpy as np
import supervision as sv
from ultralytics import YOLO

from display import show_frame, close_windows

SOURCE_VIDEO = "vehicles.mp4"
TARGET_VIDEO = "bai4_output.mp4"
VEHICLES_CLASSES= [2, 3, 5, 7]

#Load model
model = YOLO("yolov8n.pt")

#Doc thong tin video
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)

#Tao annotator
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)

def process_frame(frame: np.ndarray) -> np.ndarray:
    #Yolo phát hiện
    results = model(frame, verbose=False)[0]
    #Chuyển sang Detections
    detections = sv.Detections.from_ultralytics(results)
    #Lọc xe và confidence
    detections = detections[
        (detections.confidence > 0.3) & np.isin(detections.class_id, VEHICLES_CLASSES)
    ]
    #Tạo label
    labels = [f"{name} {conf:.2f}" for name, conf
              in zip(detections.data["class_name"], detections.confidence)]
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