import numpy as np

SOURCE_VIDEO = "vehicles.mp4"
TARGET_VIDEO = "output_final.mp4"
REPORT_PATH = "report.json"

MODEL_NAME = "yolov8n.pt"       # đổi yolov8s/m nếu có GPU
CONF_THRESHOLD = 0.3
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
SPEED_LIMIT_KMH = 80

SHOW_PREVIEW = True             # 🖥️ False nếu chạy trên server không màn hình

# Hiệu chỉnh theo video của em (Bài 6, Bài 8)
LINE_Y_RATIO = 0.5              # vạch đếm ở 50% chiều cao khung hình
PERSPECTIVE_SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
ROAD_WIDTH_M, ROAD_LENGTH_M = 25, 250