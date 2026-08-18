import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

from display import show_frame, close_windows

SOURCE_VIDEO = "vehicles.mp4"
TARGET_VIDEO = "bai7_polygonzone.mp4"
VEHICLE_CLASSES = [2, 3, 5, 7]

model = YOLO("yolov8n.pt")
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)
W, H = video_info.width, video_info.height

# Tạo ByteTrack
tracker = sv.ByteTrack(
    frame_rate=video_info.fps,
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
)

# Làm mượt kết quả tracking
smoother = sv.DetectionsSmoother(length=5)

# Tạo 2 polygon bám theo phối cảnh 2 làn đường (chia bởi dải phân cách)
polygon_left = np.array([
    [int(W * 0.36), int(H * 0.40)],
    [int(W * 0.49), int(H * 0.40)],
    [int(W * 0.53), int(H * 0.97)],
    [int(W * 0.05), int(H * 0.97)],
])

polygon_right = np.array([
    [int(W * 0.49), int(H * 0.40)],
    [int(W * 0.62), int(H * 0.40)],
    [int(W * 0.98), int(H * 0.97)],
    [int(W * 0.53), int(H * 0.97)],
])

zone_left = sv.PolygonZone(
    polygon=polygon_left,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,),
)
zone_right = sv.PolygonZone(
    polygon=polygon_right,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,),
)
zones = [zone_left, zone_right]

zone_annotator_left = sv.PolygonZoneAnnotator(
    zone=zone_left,
    color=sv.Color.RED,
    thickness=2,
    text_scale=1,
    opacity=0.15,
    display_in_zone_count=False,   # tự vẽ số ở giữa vùng bên dưới
)
zone_annotator_right = sv.PolygonZoneAnnotator(
    zone=zone_right,
    color=sv.Color.BLUE,
    thickness=2,
    text_scale=1,
    opacity=0.15,
    display_in_zone_count=False,
)
zone_annotators = [zone_annotator_left, zone_annotator_right]

# Annotator
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER,
)

# Lưu số frame ở trong vùng
frames_in_zone = defaultdict(int)


def draw_zone_count(frame, zone, polygon, color):
    # Vẽ số xe hiện tại to, đẹp, ngay giữa vùng
    cx = int(np.mean(polygon[:, 0]))
    cy = int(np.mean(polygon[:, 1]))
    text = str(zone.current_count)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
    cv2.circle(frame, (cx, cy), max(tw, th) // 2 + 25, color, -1)
    cv2.circle(frame, (cx, cy), max(tw, th) // 2 + 25, (255, 255, 255), 2)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return frame


def check_dwell(frame, detections, zones, fps):
    warnings = []
    for zone in zones:
        in_zone = zone.trigger(detections)
        for tid in detections.tracker_id[in_zone]:
            frames_in_zone[tid] += 1
            if frames_in_zone[tid] > fps * 10:
                warnings.append(int(tid))

    if warnings:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 255), -1)
        cv2.putText(frame, f"!!! XE DUNG QUA LAU TRONG VUNG: {warnings} !!!",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        print(f"Xe {warnings} dung qua lau trong vung!")

    return frame


def process_frame(frame):
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]

    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    detections.data["class_name"] = np.array([
        model.names[int(class_id)] for class_id in detections.class_id
    ])

    labels = [
        f"#{tid} {model.names[class_id]}"
        for tid, class_id in zip(detections.tracker_id, detections.class_id)
    ]

    annotated = frame.copy()
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    annotated = trace_annotator.annotate(annotated, detections)

    # Vẽ 2 vùng + số xe hiện tại ngay giữa mỗi vùng
    for zone, zone_annotator, polygon in zip(
            zones, zone_annotators, [polygon_left, polygon_right]):
        zone.trigger(detections)
        annotated = zone_annotator.annotate(annotated)
        annotated = draw_zone_count(annotated, zone, polygon, zone_annotator.color.as_bgr())

    annotated = check_dwell(annotated, detections, zones, video_info.fps)

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