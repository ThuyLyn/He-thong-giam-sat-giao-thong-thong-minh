import cv2
import supervision as sv
from ultralytics import YOLO

from display import show_frame, close_windows

#Load model yolo
model = YOLO("yolov8n.pt")

# Lấy frame từ video
image = next(
    sv.get_video_frames_generator("vehicles.mp4")
)
# Chay yolo
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

# Khởi tạo annotator (tạo 1 lần, dùng lại nhiều lần)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    text_position=sv.Position.TOP_LEFT,
)
#Tao label
labels = [
    f"{class_name} {conf:.2f}"
    for class_name, conf
    in zip(detections.data["class_name"], detections.confidence)
]

# Vẽ (luôn copy để giữ ảnh gốc)
annotated = image.copy()
annotated = box_annotator.annotate(scene=annotated, detections=detections)
annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

#  Hiện lên cửa sổ xem ngay (bấm phím bất kỳ để đóng)
show_frame(annotated, wait=0)
close_windows()

annotators = {
    "RoundBox": sv.RoundBoxAnnotator(),          # Box bo góc — nhìn hiện đại
    "BoxCorner": sv.BoxCornerAnnotator(),        # Chỉ vẽ 4 góc — phong cách "quân sự"
    "Ellipse": sv.EllipseAnnotator(),            # Ellipse dưới chân — phân tích bóng đá
    "Circle": sv.CircleAnnotator(),              # Vòng tròn bao quanh
    "Dot": sv.DotAnnotator(),                    # Chấm tại tâm
    "Triangle": sv.TriangleAnnotator(),          # Tam giác trên đầu — kiểu game
    "Color": sv.ColorAnnotator(opacity=0.4),     # Tô màu cả vùng box
    "Blur": sv.BlurAnnotator(),                  # Làm mờ đối tượng (che biển số, mặt!)
    "Pixelate": sv.PixelateAnnotator(),          # Pixel hóa đối tượng
}

for name, annotator in annotators.items():
    annotated = annotator.annotate(image.copy(), detections)
    cv2.putText(annotated, name, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    if not show_frame(annotated, window_name="So sanh Annotator", wait=0):
        break   # bấm Q thì dừng duyệt

close_windows()

# Bảng màu tùy chỉnh theo class
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(["#ff0000", "#00ff00", "#0000ff", "#ffff00"]),
    color_lookup=sv.ColorLookup.CLASS,   # màu theo class
)