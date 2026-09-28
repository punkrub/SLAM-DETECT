"""Color and shape detection for wall signs, with conservative shape scoring.

Perspective can make a square look rectangular.  This detector therefore
returns a shape only when the contour has enough rectangular/circular geometry;
ambiguous quadrilaterals are reported as ``Unknown`` so they are not written to
the mission map as a confident target.
"""

import math

import cv2
import numpy as np


BGR_COLORS = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
}

COLOR_RANGES = {
    "Red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "Green": [(np.array([40, 45, 30]), np.array([90, 255, 255]))],
    "Blue": [(np.array([100, 100, 35]), np.array([140, 255, 255]))],
    "Yellow": [(np.array([20, 90, 70]), np.array([35, 255, 255]))],
}

MIN_CONTOUR_AREA = 1000
MAX_FRAME_AREA_RATIO = 0.20
MIN_SHAPE_CONFIDENCE = 0.78
SQUARE_MAX_ASPECT = 1.28
RECT_MIN_ASPECT = 1.55
MAX_RECT_ASPECT = 3.0


def _angle_degrees(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-9:
        return 0.0
    cosine = float(np.dot(a, b) / denominator)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _quadrilateral_quality(points):
    """Score how close a four-corner contour is to a front-facing rectangle."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-6):
        return 0.0

    angles = [
        _angle_degrees(points[(i - 1) % 4] - points[i],
                       points[(i + 1) % 4] - points[i])
        for i in range(4)
    ]
    right_angle_error = float(np.mean([abs(angle - 90.0) for angle in angles]))
    opposite_length_error = 0.5 * (
        abs(lengths[0] - lengths[2]) / max(lengths[0], lengths[2])
        + abs(lengths[1] - lengths[3]) / max(lengths[1], lengths[3])
    )
    angle_score = max(0.0, 1.0 - right_angle_error / 45.0)
    side_score = max(0.0, 1.0 - opposite_length_error / 0.45)
    return 0.65 * angle_score + 0.35 * side_score


def classify_contour(contour, frame_area):
    """Return (shape, box, confidence, geometry quality), or None if noise."""
    area = float(cv2.contourArea(contour))
    if area <= MIN_CONTOUR_AREA or area > frame_area * MAX_FRAME_AREA_RATIO:
        return None

    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return None
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    x, y, width, height = cv2.boundingRect(approx)
    if width < 1 or height < 1:
        return None
    rectangularity = min(1.0, area / float(width * height))

    if len(approx) == 4 and cv2.isContourConvex(approx):
        # Use the image-aligned bounding box for horizontal/vertical labels.
        # minAreaRect's width/height can swap when OpenCV normalizes its angle.
        aspect = max(width, height) / float(min(width, height))
        quality = _quadrilateral_quality(approx)

        if aspect <= SQUARE_MAX_ASPECT:
            shape = "Square"
            aspect_score = max(0.0, 1.0 - abs(aspect - 1.0) / 0.35)
        elif RECT_MIN_ASPECT <= aspect <= MAX_RECT_ASPECT:
            shape = "Horizontal_Rect" if width >= height else "Vertical_Rect"
            expected = 2.0
            aspect_score = max(0.0, 1.0 - abs(aspect - expected) / 1.2)
        else:
            # The boundary is deliberately conservative: a skewed square may
            # land here, so do not force it into a rectangle class.
            shape = "Unknown"
            aspect_score = 0.0

        confidence = min(1.0, 0.45 * quality + 0.35 * aspect_score
                         + 0.20 * rectangularity)
        if quality < 0.65 or confidence < MIN_SHAPE_CONFIDENCE:
            shape = "Unknown"
        return shape, (x, y, width, height), confidence, quality

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    aspect = width / float(height)
    if circularity < 0.72 or not 0.78 <= aspect <= 1.28:
        return None
    confidence = min(1.0, 0.70 * circularity + 0.30 * rectangularity)
    shape = "Circle" if confidence >= MIN_SHAPE_CONFIDENCE else "Unknown"
    return shape, (x, y, width, height), confidence, circularity


def detect_signs(frame):
    """Return annotated frame, color mask, and detections compatible with caller."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR camera frame")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    overlay = frame.copy()
    mask_view = np.zeros_like(frame)
    detections = []
    frame_area = frame.shape[0] * frame.shape[1]
    kernel = np.ones((5, 5), dtype=np.uint8)

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        # Closing reconnects edges broken by glare; opening removes small noise.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            result = classify_contour(contour, frame_area)
            if result is None:
                continue
            shape, box, confidence, geometry_quality = result
            x, y, width, height = box
            detection = {
                "color": color_name,
                "shape": shape,
                "center": (x + width // 2, y + height // 2),
                "center_norm": ((x + width / 2) / frame.shape[1],
                                (y + height / 2) / frame.shape[0]),
                "contour": contour,
                "box": box,
                "shape_confidence": round(confidence, 3),
                "geometry_quality": round(geometry_quality, 3),
            }
            detections.append(detection)

            display_color = BGR_COLORS[color_name]
            cv2.drawContours(mask_view, [contour], -1, display_color, -1)
            cv2.drawContours(overlay, [contour], -1, display_color, 2)
            label = "{} {} {:.0f}%".format(
                color_name, shape, confidence * 100.0
            )
            cv2.putText(overlay, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        2, cv2.LINE_AA)

    return overlay, mask_view, detections
