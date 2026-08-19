import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque

from display import show_frame, close_windows

SOURCE_VIDEO = "vehicles.mp4"
TARGET_VIDEO = "bai8_speed.mp4"
VEHICLE_CLASSES = [2, 3, 5, 7]

model = YOLO("yolov8n.pt")
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)

# Tạo ByteTrack
tracker = sv.ByteTrack(
    frame_rate=video_info.fps,
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
)

# Làm mượt kết quả tracking
smoother = sv.DetectionsSmoother(length=5)

# 4 điểm hình thang trên ảnh (đo bằng tool click chuột theo video thực tế)
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

# Kích thước thật của vùng đó (mét)
TARGET_WIDTH, TARGET_HEIGHT = 25, 250
TARGET = np.array([
    [0, 0], [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1],
])


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(reshaped, self.m).reshape(-1, 2)


view_transformer = ViewTransformer(SOURCE, TARGET)

# Annotator
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER,
)

# Lưu lịch sử tọa độ y (mét) của từng track trong 1 giây gần nhất
coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))


def compute_speed_labels(detections):
    points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    points = view_transformer.transform_points(points)

    labels = []
    for tracker_id, (_, y) in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)
        if len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            # quãng đường (m) đi được trong khoảng thời gian (s)
            distance = abs(coordinates[tracker_id][-1] - coordinates[tracker_id][0])
            time_s = len(coordinates[tracker_id]) / video_info.fps
            speed_kmh = distance / time_s * 3.6
            labels.append(f"#{tracker_id} {int(speed_kmh)} km/h")
    return labels


def process_frame(frame):
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]

    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    labels = compute_speed_labels(detections)

    annotated = frame.copy()
    annotated = sv.draw_polygon(annotated, polygon=SOURCE, color=sv.Color.RED, thickness=4)
    annotated = trace_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)

    return annotated


with sv.VideoSink(target_path=TARGET_VIDEO, video_info=video_info) as sink:
    for frame in sv.get_video_frames_generator(SOURCE_VIDEO):
        annotated = process_frame(frame)
        sink.write_frame(annotated)
        if not show_frame(annotated):
            print("Nguoi dung bam Q - dung som.")
            break

close_windows()
print("Xong! Video da luu tai:", TARGET_VIDEO)