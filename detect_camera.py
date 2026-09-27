import cv2
import numpy as np
from robomaster import robot

BGR_COLORS = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
}


MIN_CONTOUR_AREA = 1000
COLOR_RANGES = {
    "Red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "Green": [(np.array([40, 45, 30]), np.array([90, 255, 255]))],
    "Blue": [(np.array([100, 100, 35]), np.array([140, 255, 255]))],
    "Yellow": [(np.array([20, 90, 70]), np.array([35, 255, 255]))],
}
MIN_SIZE_CLUSTER_COUNT = 3
MIN_SIZE_RATIO = 0.55
MAX_SIZE_RATIO = 1.8


def classify_sign(contour, frame_shape, frame_area):
    area = cv2.contourArea(contour)
    if area <= MIN_CONTOUR_AREA or area > frame_area * 0.2:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    if not cv2.isContourConvex(approx):
        return None

    x, y, width, height = cv2.boundingRect(approx)
    frame_height, frame_width = frame_shape
    if width > frame_width * 0.75 or height > frame_height * 0.75:
        return None

    aspect_ratio = width / float(height)
    rectangularity = area / float(width * height)
    if len(approx) == 4 and rectangularity >= 0.75:
        if 0.55 <= aspect_ratio <= 1.8:
            shape_name = "Square"
        elif 1.8 < aspect_ratio <= 3.0:
            shape_name = "Horizontal_Rect"
        elif 0.35 <= aspect_ratio < 0.55:
            shape_name = "Vertical_Rect"
        else:
            return None
    else:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity <= 0.78 or not 0.75 <= aspect_ratio <= 1.33:
            return None
        shape_name = "Circle"

    return shape_name, (x, y, width, height)


def filter_size_outliers(detections):
    if len(detections) < MIN_SIZE_CLUSTER_COUNT:
        return detections

    scales = [
        np.sqrt(width * height)
        for _, _, width, height in (item["box"] for item in detections)
    ]
    median_scale = float(np.median(scales))
    return [
        item for item, scale in zip(detections, scales)
        if MIN_SIZE_RATIO * median_scale <= scale <= MAX_SIZE_RATIO * median_scale
    ]


def detect_signs(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    overlay = frame.copy()
    mask_view = np.zeros_like(frame)
    detections = []
    frame_area = frame.shape[0] * frame.shape[1]

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower, upper))

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            classification = classify_sign(contour, frame.shape[:2], frame_area)
            if classification is None:
                continue
            shape_name, (x, y, width, height) = classification

            center = (x + width // 2, y + height // 2)
            detections.append({
                "color": color_name,
                "shape": shape_name,
                "center": center,
                "contour": contour,
                "box": (x, y, width, height),
            })

    detections = filter_size_outliers(detections)
    for detection in detections:
        color_name = detection["color"]
        shape_name = detection["shape"]
        contour = detection["contour"]
        x, y, width, height = detection["box"]
        cv2.drawContours(mask_view, [contour], -1, BGR_COLORS[color_name], -1)
        cv2.drawContours(overlay, [contour], -1, BGR_COLORS[color_name], -1)
        cv2.drawContours(overlay, [contour], -1, (255, 255, 255), 2)
        label = "{} {}".format(color_name, shape_name)
        label_y = max(y - 8, 20)
        cv2.putText(
            overlay, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, overlay)
    return overlay, mask_view, detections


def main():
    ep_robot = robot.Robot()
    ep_camera = None
    stream_started = False

    try:
        print("Connecting to robot camera via AP...")
        ep_robot.initialize(conn_type="ap")
        ep_camera = ep_robot.camera
        ep_camera.start_video_stream(display=False)
        stream_started = True
        print("Camera detection started. Press 'q' to quit.")

        previous_signs = None
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is not None:
                result, mask_view, detections = detect_signs(frame)
                signs = tuple(sorted(
                    "{} {}".format(item["color"], item["shape"])
                    for item in detections
                ))

                if signs != previous_signs:
                    if signs:
                        print("Detected {} sign(s): {}".format(
                            len(signs), ", ".join(signs)
                        ))
                    else:
                        print("No recognized signs in view.")
                    previous_signs = signs

                cv2.imshow("RoboMaster - Sign Detection", result)
                cv2.imshow("Color Mask", mask_view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as error:
        print("Error: {}".format(error))
    finally:
        cv2.destroyAllWindows()
        if stream_started:
            ep_camera.stop_video_stream()
        ep_robot.close()
        print("Camera disconnected.")


if __name__ == "__main__":
    main()