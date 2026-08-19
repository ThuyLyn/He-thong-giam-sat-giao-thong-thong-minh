import json
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import config
from display import show_frame, close_windows
from transformer import ViewTransformer


class TrafficMonitor:
    def __init__(self):
        self.model = YOLO(config.MODEL_NAME)
        self.video_info = sv.VideoInfo.from_video_path(config.SOURCE_VIDEO)
        fps = self.video_info.fps

        # --- Tracking (Bài 5) ---
        self.tracker = sv.ByteTrack(frame_rate=fps)
        self.smoother = sv.DetectionsSmoother(length=5)

        # --- Đếm qua vạch (Bài 6) ---
        y = int(self.video_info.height * config.LINE_Y_RATIO)
        self.line_zone = sv.LineZone(
            start=sv.Point(0, y),
            end=sv.Point(self.video_info.width, y),
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
            minimum_crossing_threshold=2,
        )
        self.class_counts = defaultdict(int)

        # --- Đo tốc độ (Bài 8) ---
        target = np.array([
            [0, 0], [config.ROAD_WIDTH_M - 1, 0],
            [config.ROAD_WIDTH_M - 1, config.ROAD_LENGTH_M - 1],
            [0, config.ROAD_LENGTH_M - 1],
        ])
        self.view_transformer = ViewTransformer(config.PERSPECTIVE_SOURCE, target)
        self.coordinates = defaultdict(lambda: deque(maxlen=int(fps)))
        self.speed_violations = {}   # tracker_id -> max speed vi phạm

        # --- Annotators (Bài 2) ---
        thickness = sv.calculate_optimal_line_thickness(self.video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(self.video_info.resolution_wh)
        self.box_annotator = sv.BoxAnnotator(
            thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale, color_lookup=sv.ColorLookup.TRACK)
        self.trace_annotator = sv.TraceAnnotator(
            trace_length=int(fps * 2), thickness=thickness,
            color_lookup=sv.ColorLookup.TRACK)
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=thickness, text_scale=text_scale,
            custom_in_text="Vao", custom_out_text="Ra")

    # ---------- các bước pipeline ----------

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        mask = (detections.confidence > config.CONF_THRESHOLD) & \
               np.isin(detections.class_id, config.VEHICLE_CLASSES)
        return detections[mask]

    def track(self, detections):
        detections = self.tracker.update_with_detections(detections)
        return self.smoother.update_with_detections(detections)

    def count(self, detections):
        crossed_in, crossed_out = self.line_zone.trigger(detections)
        for name in detections.data["class_name"][crossed_in]:
            self.class_counts[f"{name}_in"] += 1
        for name in detections.data["class_name"][crossed_out]:
            self.class_counts[f"{name}_out"] += 1

    def speeds(self, detections):
        pts = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pts = self.view_transformer.transform_points(pts)
        fps = self.video_info.fps

        labels = []
        for tid, name, (_, y) in zip(
                detections.tracker_id, detections.data["class_name"], pts):
            self.coordinates[tid].append(y)
            if len(self.coordinates[tid]) < fps / 2:
                labels.append(f"#{tid} {name}")
                continue
            dist = abs(self.coordinates[tid][-1] - self.coordinates[tid][0])
            speed = dist / (len(self.coordinates[tid]) / fps) * 3.6
            tag = " !QUA TOC DO!" if speed > config.SPEED_LIMIT_KMH else ""
            if tag:
                self.speed_violations[int(tid)] = max(
                    self.speed_violations.get(int(tid), 0), int(speed))
            labels.append(f"#{tid} {name} {int(speed)}km/h{tag}")
        return labels

    def draw_dashboard(self, frame):
        """🖥️ Bảng thống kê realtime góc trên trái — nhìn phát biết ngay tình hình."""
        stats = {
            "Tong VAO": self.line_zone.in_count,
            "Tong RA": self.line_zone.out_count,
            "Vi pham toc do": len(self.speed_violations),
            **dict(self.class_counts),
        }
        x, y, line_h = 25, 70, 55
        h = line_h * len(stats) + 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (560, 10 + h), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        for i, (k, v) in enumerate(stats.items()):
            color = (0, 0, 255) if "Vi pham" in k and v > 0 else (255, 255, 255)
            cv2.putText(frame, f"{k}: {v}", (x, y + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        return frame

    def annotate(self, frame, detections, labels):
        out = frame.copy()
        out = self.trace_annotator.annotate(out, detections)
        out = self.box_annotator.annotate(out, detections)
        out = self.label_annotator.annotate(out, detections, labels=labels)
        out = self.line_annotator.annotate(out, line_counter=self.line_zone)
        out = self.draw_dashboard(out)          # 🖥️ bảng thống kê realtime
        return out

    # ---------- chạy ----------

    def process_frame(self, frame):
        detections = self.detect(frame)
        detections = self.track(detections)
        self.count(detections)
        labels = self.speeds(detections)
        return self.annotate(frame, detections, labels)

    def run(self):
        # 🖥️ Vòng lặp bộ khung Bài 4: XEM LIVE + GHI FILE song song, Q để dừng
        with sv.VideoSink(target_path=config.TARGET_VIDEO,
                          video_info=self.video_info) as sink:
            for frame in sv.get_video_frames_generator(config.SOURCE_VIDEO):
                annotated = self.process_frame(frame)
                sink.write_frame(annotated)
                if config.SHOW_PREVIEW and not show_frame(
                        annotated, window_name="Traffic Monitor - Q de thoat"):
                    print("Da dung theo yeu cau nguoi dung.")
                    break
        close_windows()
        self.export_report()

    def export_report(self):
        report = {
            "video": config.SOURCE_VIDEO,
            "total_in": self.line_zone.in_count,
            "total_out": self.line_zone.out_count,
            "by_class": dict(self.class_counts),
            "speed_violations": self.speed_violations,
        }
        with open(config.REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))