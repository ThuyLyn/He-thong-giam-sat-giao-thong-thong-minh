import cv2

WINDOW_NAME = "Supervision - Live"
MAX_DISPLAY_WIDTH = 1280   # thu nhỏ frame cho vừa màn hình (chỉ để XEM, không ảnh hưởng xử lý)


def show_frame(frame, window_name: str = WINDOW_NAME, wait: int = 1) -> bool:
    """Hiện frame lên cửa sổ. Trả về False nếu người dùng bấm Q/ESC (muốn thoát).

    wait=1  -> dùng cho video (hiện liên tục, không chặn)
    wait=0  -> dùng cho ảnh tĩnh (dừng lại chờ bấm phím bất kỳ)
    """
    h, w = frame.shape[:2]
    if w > MAX_DISPLAY_WIDTH:                      # thu nhỏ để vừa màn hình
        scale = MAX_DISPLAY_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(wait) & 0xFF
    if key in (ord("q"), ord("Q"), 27):            # Q hoặc ESC -> thoát
        return False
    return True


def close_windows():
    cv2.destroyAllWindows()