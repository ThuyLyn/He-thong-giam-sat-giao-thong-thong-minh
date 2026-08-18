import supervision as sv

from display import show_frame, close_windows


# Lấy frame đầu tiên của video
frame = next(
    sv.get_video_frames_generator("vehicles.mp4")
)

# Hiển thị frame
show_frame(frame, wait=0)

# Đóng cửa sổ
close_windows()