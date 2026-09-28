import argparse
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from sign_vision import MIN_SHAPE_CONFIDENCE, detect_signs
from SLAM.src.grid_slam import DIRECTIONS, NAMES, DFSExplorer, wrap
from SLAM.src.settings import get as setting
from SLAM.src.settings import project_path


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_SENSOR_AGE_SEC = setting("slam.max_sensor_age_sec")
FRONT_VIEW_TOLERANCE_DEG = 12.0
MAX_STATIONARY_SPEED_MPS = 0.03
SIGN_CONFIRMATION_FRAMES = 3
MAX_TOF_MARK_SAMPLES = 15
SIGN_LOOK_DOWN_PITCH_DEG = -20.0
SIGN_INSPECTION_TIMEOUT_SEC = 3.0
SIGN_CAMERA_SWEEP_OFFSETS_DEG = (-8.0, 0.0, 8.0)
FRONT_STOP_TARGET_MM = 220.0


def front_wall_measurement(slam, state, now=None):
    if now is None:
        now = time.monotonic()

    if not state.tof_valid or state.tof_filtered_mm is None:
        return None
    if not 0 < state.tof_received_at <= now:
        return None
    if now - state.tof_received_at > MAX_SENSOR_AGE_SEC:
        return None
    if not 0 < state.gimbal_received_at <= now:
        return None
    if now - state.gimbal_received_at > MAX_SENSOR_AGE_SEC:
        return None
    if abs(state.gimbal_yaw) > FRONT_VIEW_TOLERANCE_DEG:
        return None
    chassis_heading_error = wrap(state.yaw - wrap(slam.heading * 90))
    if abs(chassis_heading_error) > FRONT_VIEW_TOLERANCE_DEG:
        return None
    if not state.is_static or any(
        abs(speed) > MAX_STATIONARY_SPEED_MPS
        for speed in (state.vel_vx, state.vel_vy, state.vel_vz)
    ):
        return None

    position_error = math.hypot(
        state.pos_x - slam.pose[0], state.pos_y - slam.pose[1]
    )
    if position_error > setting("slam.localization_gate_m") + 0.05:
        return None

    return front_wall_distance(slam, slam.heading, state.tof_filtered_mm / 1000.0)


def front_wall_distance(slam, direction, distance_m, sensor_heading=None):
    if direction not in range(4) or not math.isfinite(distance_m) or distance_m <= 0:
        return None
    if sensor_heading is None:
        sensor_heading = direction

    dx, dy = DIRECTIONS[direction]
    axis, sign = (0, dx) if dx else (1, dy)
    offset = slam.offset(sensor_heading, direction)
    boundary = (slam.cell[axis] + sign * 0.5) * setting("navigation.grid_size_m")
    expected_mm = 1000 * (
        sign * (boundary - slam.pose[axis] - offset[axis])
        - setting("slam.wall_thickness_m") / 2
    )
    if expected_mm <= 0:
        return None

    distance_mm = distance_m * 1000
    if distance_mm > expected_mm + setting("slam.wall_margin_m") * 1000:
        return None
    return distance_mm


def inspect_walls_during_scan(explorer, inspection, lock, finished):
    backend = explorer.backend
    original_scan = backend.scan
    robot = backend.robot

    def scan_with_sign_inspection(*args, **kwargs):
        scan_result = original_scan(*args, **kwargs)
        ranges, heading, _ = scan_result
        scan_headings = getattr(backend, "scan_headings", None) or {}
        walls = []
        for direction, distance_m in ranges.items():
            sensor_heading = scan_headings.get(direction, heading)
            distance_mm = front_wall_distance(
                explorer.slam, direction, distance_m, sensor_heading
            )
            if distance_mm is not None:
                walls.append((direction, sensor_heading, distance_mm))

        state = backend.hub.get_latest_state()
        if (any(abs(speed) > MAX_STATIONARY_SPEED_MPS
                for speed in (state.vel_vx, state.vel_vy, state.vel_vz))
                or abs(wrap(state.yaw - wrap(heading * 90))) > FRONT_VIEW_TOLERANCE_DEG):
            return scan_result

        yaw_for_relative_direction = {0: 0, 1: 90, 2: -180, 3: -90}
        for direction, sensor_heading, distance_mm in walls:
            relative_direction = (direction - heading) % 4
            camera_yaw = yaw_for_relative_direction[relative_direction]
            finished.clear()
            try:
                action = robot.gimbal.moveto(
                    pitch=SIGN_LOOK_DOWN_PITCH_DEG,
                    yaw=camera_yaw,
                    pitch_speed=setting("gimbal.pitch_speed_dps"),
                    yaw_speed=setting("gimbal.yaw_speed_dps"),
                )
                completed = action.wait_for_completed(
                    timeout=setting("gimbal.action_timeout_sec")
                )
                if completed is False or getattr(action, "has_succeeded", True) is False:
                    print("Sign look-down action failed for {} wall.".format(NAMES[direction]))
                    continue

                with lock:
                    inspection["inspection_id"] = inspection.get("inspection_id", 0) + 1
                    inspection.update({
                        "active": True,
                        "cell": tuple(explorer.slam.cell),
                        "direction": direction,
                        "tof_distance_mm": distance_mm,
                        "frame_count": 0,
                        "view_offset": 0.0,
                    })

                sweep_yaws = [
                    camera_yaw + offset
                    for offset in SIGN_CAMERA_SWEEP_OFFSETS_DEG
                    if -250 <= camera_yaw + offset <= 250
                ]
                print("Inspecting {} wall in cell {} at yaw {} for {:.1f}s...".format(
                    NAMES[direction], tuple(explorer.slam.cell),
                    sweep_yaws, SIGN_INSPECTION_TIMEOUT_SEC,
                ))
                deadline = time.monotonic() + SIGN_INSPECTION_TIMEOUT_SEC
                dwell_per_yaw = SIGN_INSPECTION_TIMEOUT_SEC / max(1, len(sweep_yaws))
                for sweep_yaw in sweep_yaws:
                    if finished.is_set() or not backend.controller.is_running():
                        break
                    with lock:
                        inspection["active"] = False
                    try:
                        sweep_action = robot.gimbal.moveto(
                            pitch=SIGN_LOOK_DOWN_PITCH_DEG,
                            yaw=sweep_yaw,
                            pitch_speed=setting("gimbal.pitch_speed_dps"),
                            yaw_speed=setting("gimbal.yaw_speed_dps"),
                        )
                        sweep_completed = sweep_action.wait_for_completed(
                            timeout=setting("gimbal.action_timeout_sec")
                        )
                        if (sweep_completed is False
                                or getattr(sweep_action, "has_succeeded", True) is False):
                            continue
                        time.sleep(setting("gimbal.settle_sec"))
                        with lock:
                            inspection["cell"] = tuple(explorer.slam.cell)
                            inspection["direction"] = direction
                            inspection["tof_distance_mm"] = distance_mm
                            inspection["camera_yaw"] = sweep_yaw
                            inspection["view_offset"] = sweep_yaw - camera_yaw
                            inspection["active"] = True
                        dwell_deadline = min(
                            deadline, time.monotonic() + dwell_per_yaw
                        )
                        while time.monotonic() < dwell_deadline:
                            if finished.wait(min(0.02, dwell_deadline - time.monotonic())):
                                break
                    except Exception as error:
                        print("Gimbal sweep warning ({} deg): {}".format(
                            sweep_yaw, error
                        ))
            except Exception as error:
                print("Sign look-down warning ({}): {}".format(NAMES[direction], error))
            finally:
                with lock:
                    inspection["active"] = False
                try:
                    restore_action = robot.gimbal.moveto(
                        pitch=setting("gimbal.pitch_deg"),
                        yaw=0,
                        pitch_speed=setting("gimbal.pitch_speed_dps"),
                        yaw_speed=setting("gimbal.yaw_speed_dps"),
                    )
                    restored = restore_action.wait_for_completed(
                        timeout=setting("gimbal.action_timeout_sec")
                    )
                    if restored is False or getattr(restore_action, "has_succeeded", True) is False:
                        print("Warning: could not restore the configured gimbal pose.")
                except Exception as error:
                    print("Gimbal restore warning: {}".format(error))

        return scan_result

    backend.scan = scan_with_sign_inspection


def save_sign_marks(output, marks):
    path = Path(output)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["map_info"]["rows"]
    columns = data["map_info"]["columns"]
    visited = {tuple(cell) for cell in data.get("visited", [])}
    known_edges = {
        tuple(sorted(tuple(cell) for cell in edge["cells"])): edge["wall"]
        for edge in data.get("edges", [])
    }
    for mark in marks.values():
        cell = tuple(mark["cell"])
        direction = NAMES.index(mark["direction"])
        dx, dy = DIRECTIONS[direction]
        # The observed cell is one tile from the wall sign. The next cell
        # behind the robot is the two-tile shooting position.
        candidates = [(1, cell), (2, (cell[0] - dx, cell[1] - dy))]
        mark["shooting_positions"] = [
            {"distance_tiles": distance, "cell": list(position),
             "facing": mark["direction"]}
            for distance, position in candidates
            if 0 <= position[0] < rows and 0 <= position[1] < columns
            and position in visited
            and (distance == 1 or known_edges.get(tuple(sorted((cell, position)))) is False)
        ]
    data["signs"] = sorted(
        marks.values(),
        key=lambda item: (item["cell"][0], item["cell"][1], item["direction"],
                          item["color"], item["shape"]),
    )
    data["sign_count"] = len(data["signs"])
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def assign_sign_instance(instance_tracks, cell, direction, detection):
    """Associate detections by color and image position so equal signs stay distinct."""
    color_key = (tuple(cell), direction, detection["color"])
    tracks = instance_tracks.setdefault(color_key, [])
    center = detection["center_norm"]
    best = None
    best_distance = float("inf")
    for track in tracks:
        distance = math.hypot(center[0] - track["center"][0],
                              center[1] - track["center"][1])
        if distance < best_distance:
            best, best_distance = track, distance
    if best is None or best_distance > 0.20:
        instance_id = len(tracks)
        tracks.append({"id": instance_id, "center": tuple(center), "observations": 1})
        return instance_id
    count = best["observations"]
    best["center"] = (
        (best["center"][0] * count + center[0]) / (count + 1),
        (best["center"][1] * count + center[1]) / (count + 1),
    )
    best["observations"] = count + 1
    return best["id"]


def render_map_image(map_file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, Rectangle

    path = Path(map_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    cell_size = data["cell_size_m"]
    rows = data["map_info"]["rows"]
    columns = data["map_info"]["columns"]
    figure, axis = plt.subplots(figsize=(10, 8))

    for cell_x, cell_y in data["visited"]:
        axis.add_patch(Rectangle(
            (cell_y * cell_size - cell_size / 2,
             cell_x * cell_size - cell_size / 2),
            cell_size, cell_size, facecolor="#e7f1f7",
            edgecolor="#aab9c4", linewidth=0.8, zorder=0,
        ))

    for edge in data["edges"]:
        if not edge["wall"]:
            continue
        (cell_a, cell_b) = edge["cells"]
        if cell_a[0] != cell_b[0]:
            wall_x = (cell_a[1] + cell_b[1]) * cell_size / 2
            wall_y = (cell_a[0] + cell_b[0]) * cell_size / 2
            axis.plot([wall_x - cell_size / 2, wall_x + cell_size / 2],
                      [wall_y, wall_y], color="#263746", linewidth=3.2,
                      solid_capstyle="round", zorder=2)
        else:
            wall_x = (cell_a[1] + cell_b[1]) * cell_size / 2
            wall_y = (cell_a[0] + cell_b[0]) * cell_size / 2
            axis.plot([wall_x, wall_x],
                      [wall_y - cell_size / 2, wall_y + cell_size / 2],
                      color="#263746", linewidth=3.2,
                      solid_capstyle="round", zorder=2)

    trajectory = data.get("trajectory", [])
    if trajectory:
        axis.plot(
            [point["pose"][1] for point in trajectory],
            [point["pose"][0] for point in trajectory],
            "o-", color="#1486bd", markersize=3.5, linewidth=1.5,
            label="Robot path", zorder=3,
        )

    start = data.get("start_pose", [0.0, 0.0, 0.0])
    axis.scatter([start[1]], [start[0]], marker="s", s=105,
                 color="#23a66f", edgecolors="white", linewidths=1,
                 label="Start {}".format(tuple(data.get("start_cell", [0, 0]))),
                 zorder=6)
    axis.scatter([data["pose"][1]], [data["pose"][0]], marker="X", s=100,
                 color="#e4473d", edgecolors="white", linewidths=0.8,
                 label="Last estimated pose", zorder=7)

    palette = {
        "Red": "#e3423a", "Green": "#28a765",
        "Blue": "#2776c9", "Yellow": "#e2bd00",
    }
    marker_labels = set()
    for sign in data.get("signs", []):
        cell_x, cell_y = sign["cell"]
        direction = NAMES.index(sign["direction"])
        color = palette.get(sign["color"], "#8b5fbf")
        offset = cell_size * 0.34
        center_x, center_y = cell_y * cell_size, cell_x * cell_size
        if direction == 0:
            center_y += offset
            label_y = center_y + cell_size * 0.10
            label_x = center_x
            vertical_align = "bottom"
        elif direction == 1:
            center_x += offset
            label_x = center_x + cell_size * 0.09
            label_y = center_y
            vertical_align = "center"
        elif direction == 2:
            center_y -= offset
            label_y = center_y - cell_size * 0.10
            label_x = center_x
            vertical_align = "top"
        else:
            center_x -= offset
            label_x = center_x - cell_size * 0.09
            label_y = center_y
            vertical_align = "center"

        shape = sign["shape"]
        if shape == "Circle":
            patch = Circle((center_x, center_y), cell_size * 0.055)
        else:
            width = cell_size * (0.15 if shape == "Horizontal_Rect" else 0.10)
            height = cell_size * (0.15 if shape == "Vertical_Rect" else 0.10)
            patch = Rectangle((center_x - width / 2, center_y - height / 2),
                              width, height)
        patch.set_facecolor(color)
        patch.set_edgecolor("#14212b")
        patch.set_linewidth(1.2)
        patch.set_zorder(8)
        axis.add_patch(patch)

        label = "{} {} ({})".format(sign["color"], shape, sign["direction"])
        axis.annotate(
            label, (label_x, label_y), ha="center", va=vertical_align,
            fontsize=7.5, fontweight="bold", color="#18232c", zorder=9,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                  "edgecolor": color, "alpha": 0.96, "linewidth": 1.0},
        )
        marker_labels.add((sign["color"], shape, color))

    handles, labels = axis.get_legend_handles_labels()
    handles.insert(0, Line2D([0], [0], color="#263746", linewidth=3.2, label="Wall"))
    labels.insert(0, "Wall")
    for color_name, shape, color in sorted(marker_labels):
        handles.append(Patch(facecolor=color, edgecolor="#14212b"))
        labels.append("{} {} mark".format(color_name, shape))
    axis.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.96)

    axis.set_xlim(-cell_size / 2, (columns - 0.5) * cell_size)
    axis.set_ylim(-cell_size / 2, (rows - 0.5) * cell_size)
    axis.set_xticks([(index - 0.5) * cell_size for index in range(columns + 1)])
    axis.set_yticks([(index - 0.5) * cell_size for index in range(rows + 1)])
    axis.set_aspect("equal")
    axis.set_axisbelow(True)
    axis.grid(color="#aab9c4", linewidth=0.7, alpha=0.7)
    axis.set_xlabel("Initial right axis y (m)")
    axis.set_ylabel("Initial forward axis x (m)")
    axis.set_title(
        "Exploration map | {} | {} / {} cells visited | {} sign marks".format(
            data["status"], len(data["visited"]), rows * columns,
            len(data.get("signs", [])),
        )
    )
    figure.tight_layout()
    image_path = path.with_name("map.png")
    figure.savefig(str(image_path), dpi=180)
    plt.close(figure)
    return image_path


def run_camera_loop(explorer, camera, camera_is_robot, stop_motion=None):
    outcome = {"completed": False, "error": None}
    sign_marks = {}
    instance_tracks = {}
    confirmation_streaks = {}
    current_inspection_id = None
    inspection = {"active": False, "inspection_id": 0}
    inspection_lock = threading.Lock()
    inspection_finished = threading.Event()

    if camera_is_robot:
        inspect_walls_during_scan(
            explorer, inspection, inspection_lock, inspection_finished
        )

    def explore():
        try:
            outcome["completed"] = explorer.run()
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=explore, name="maze-explorer", daemon=True)
    worker.start()
    announced = False

    try:
        print("Camera is live. Press 'q' to stop/close the mission window.")
        while True:
            if camera_is_robot:
                frame = camera.read_cv2_image(strategy="newest", timeout=0.5)
            else:
                ok, frame = camera.read()
                if not ok:
                    frame = None

            if frame is not None:
                result, mask, detections = detect_signs(frame)
                valid_wall_distance = None
                active_inspection = None
                if camera_is_robot:
                    with inspection_lock:
                        if inspection["active"]:
                            active_inspection = dict(inspection)
                    if active_inspection is not None:
                        valid_wall_distance = active_inspection["tof_distance_mm"]

                current_cell = (
                    active_inspection["cell"] if active_inspection is not None
                    else tuple(explorer.slam.cell)
                )
                current_direction = (
                    active_inspection["direction"] if active_inspection is not None
                    else explorer.slam.heading
                )

                valid_candidates = set()
                if valid_wall_distance is not None:
                    inspection_id = active_inspection.get("inspection_id")
                    if inspection_id != current_inspection_id:
                        confirmation_streaks.clear()
                        current_inspection_id = inspection_id
                    for detection in detections:
                        if (detection["shape"] == "Unknown"
                                or detection["shape_confidence"] < MIN_SHAPE_CONFIDENCE):
                            continue
                        instance_id = assign_sign_instance(
                            instance_tracks, current_cell, current_direction, detection
                        )
                        key = (current_cell, current_direction,
                               detection["color"], detection["shape"], instance_id)
                        if key in valid_candidates:
                            continue
                        valid_candidates.add(key)
                        evidence = confirmation_streaks.setdefault(
                            key, {"frames": 0, "views": set(), "confidences": []}
                        )
                        evidence["frames"] += 1
                        evidence["views"].add(round(active_inspection.get("view_offset", 0)))
                        evidence["confidences"].append(detection["shape_confidence"])
                        confirmed = (
                            evidence["frames"] >= SIGN_CONFIRMATION_FRAMES
                            and len(evidence["views"]) >= 2
                        )
                        if confirmed:
                            mark = sign_marks.get(key)
                            if mark is None:
                                mark = {
                                    "color": detection["color"],
                                    "shape": detection["shape"],
                                    "instance_id": instance_id,
                                    "cell": list(current_cell),
                                    "direction": NAMES[current_direction],
                                    "tof_distances_mm": [],
                                    "confirmed_frames": evidence["frames"],
                                    "view_offsets_deg": sorted(evidence["views"]),
                                    "shape_confidence": round(float(np.mean(
                                        evidence["confidences"])), 3),
                                    "observation_count": 0,
                                    "image_center_norm": [],
                                }
                                sign_marks[key] = mark
                            elif evidence["frames"] > SIGN_CONFIRMATION_FRAMES:
                                mark["confirmed_frames"] += 1
                            mark["observation_count"] += 1
                            mark["shape_confidence"] = round(float(np.mean(
                                evidence["confidences"])), 3)
                            mark["view_offsets_deg"] = sorted(evidence["views"])
                            center_norm = detection.get("center_norm")
                            if center_norm is not None:
                                mark["image_center_norm"].append([
                                    round(float(center_norm[0]), 4),
                                    round(float(center_norm[1]), 4),
                                ])
                                mark["image_center_norm"] = mark["image_center_norm"][-15:]
                            mark["tof_distances_mm"].append(
                                round(valid_wall_distance, 1)
                            )
                            mark["tof_distances_mm"] = mark["tof_distances_mm"][-MAX_TOF_MARK_SAMPLES:]
                            mark["tof_distance_mm"] = round(
                                float(np.median(mark["tof_distances_mm"])), 1
                            )
                            mark["last_seen_timestamp"] = time.time()
                    confirmation_streaks = {
                        key: evidence for key, evidence in confirmation_streaks.items()
                        if key in valid_candidates
                    }

                if active_inspection is not None:
                    with inspection_lock:
                        if inspection.get("active"):
                            inspection["frame_count"] += 1

                gate_text = (
                    "WALL VERIFIED: sign {}/{} frames, {} views | marked {}"
                    .format(max((item["frames"] for item in confirmation_streaks.values()),
                                default=0),
                            SIGN_CONFIRMATION_FRAMES,
                            max((len(item["views"])
                                 for item in confirmation_streaks.values()), default=0),
                            len(sign_marks))
                    if active_inspection is not None
                    else "WAITING FOR FRONT WALL SCAN - signs are not marked"
                )
                cv2.putText(result, gate_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Maze SLAM - Sign Detection", result)
                cv2.imshow("Detected Sign Mask", mask)
                if detections:
                    labels = sorted(set(
                        "{} {}".format(item["color"], item["shape"])
                        for item in detections
                    ))
                    cv2.setWindowTitle(
                        "Maze SLAM - Sign Detection",
                        "Maze SLAM - " + ", ".join(labels),
                    )

            if not worker.is_alive() and not announced:
                status = explorer.status
                print("Maze exploration {} after {} move(s), {} cell(s) visited."
                      .format(status, explorer.moves, len(explorer.slam.map.visited)))
                print("Map saved to: {}".format(explorer.output))
                announced = True

            if cv2.waitKey(1) & 0xFF == ord("q"):
                inspection_finished.set()
                if worker.is_alive() and stop_motion is not None:
                    print("Stopping maze motion...")
                    stop_motion()
                break

            if not worker.is_alive() and outcome["error"] is not None:
                break
            if not worker.is_alive():
                break
    finally:
        inspection_finished.set()
        if worker.is_alive() and stop_motion is not None:
            stop_motion()
        worker.join()
        cv2.destroyAllWindows()
        if camera_is_robot:
            save_sign_marks(explorer.output, sign_marks)
            print("Saved {} confirmed sign mark(s) to the SLAM map."
                  .format(len(sign_marks)))

    if outcome["error"] is not None:
        raise outcome["error"]
    return outcome["completed"]


def run_camera_preview(camera_index):
    """Show the detector from a local camera without starting SLAM or motion."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Cannot open webcam/camera index {}".format(camera_index))
    try:
        print("Camera preview only. Press 'q' to close; no SLAM map will be written.")
        while True:
            ok, frame = camera.read()
            if ok and frame is not None:
                result, mask, _ = detect_signs(frame)
                cv2.imshow("Sign Detection Preview", result)
                cv2.imshow("Detected Sign Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        return True
    finally:
        camera.release()
        cv2.destroyAllWindows()


def run_hardware(conn_type, calibration_path, output):
    from SLAM.src.robot_system import RobotSystem
    from SLAM.src.slam_hardware import HardwareBackend

    system = RobotSystem(calibration_file=str(calibration_path), conn_type=conn_type)
    camera = None
    stream_started = False
    try:
        if not system.connect_robot():
            raise RuntimeError("Could not connect to RoboMaster EP; refusing to use mock mode")

        system.setup_threads()
        system.thread_2_controller.wall_pid.front_target_mm = max(
            system.thread_2_controller.wall_pid.front_target_mm,
            FRONT_STOP_TARGET_MM,
        )
        print("Front-wall stopping target: {:.0f} mm (plus configured stop tolerance)."
              .format(system.thread_2_controller.wall_pid.front_target_mm))
        system.thread_1_sensor.start_collecting()
        system.thread_2_controller.enable_motion()

        deadline = time.monotonic() + setting("slam.sensor_timeout_sec")
        while system.sensor_hub.get_latest_state().frame_index == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("No initial sensor data from RoboMaster EP")
            time.sleep(0.01)

        camera = system.robot.camera
        camera.start_video_stream(display=False)
        stream_started = True
        explorer = DFSExplorer(HardwareBackend(system), output)
        stop_motion = system.thread_2_controller.stop_running
        return run_camera_loop(
            explorer, camera, camera_is_robot=True, stop_motion=stop_motion
        )
    finally:
        if system.thread_2_controller is not None:
            try:
                system.thread_2_controller.stop_running()
            except Exception as error:
                print("Motion stop warning: {}".format(error))
        try:
            if stream_started:
                camera.stop_video_stream()
        finally:
            system.shutdown(run_analysis=False)


def main():
    parser = argparse.ArgumentParser(
        description="Explore a maze with Grid SLAM while detecting signs from the camera"
    )
    parser.add_argument("--mock", action="store_true",
                        help="Preview sign detection from a PC webcam (no SLAM or motion)")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="PC webcam index used with --mock")
    parser.add_argument("--conn-type", choices=("ap", "sta"),
                        default=setting("robot.conn_type"))
    parser.add_argument("--calibration", default=project_path("paths.calibration"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "mission_maps"),
                        help="Folder for timestamped mission results (outside SLAM)")
    parser.add_argument("--output", default=None,
                        help="Optional explicit map JSON path; overrides --output-dir")
    args = parser.parse_args()

    calibration_path = Path(args.calibration)
    if not calibration_path.is_absolute():
        calibration_path = PROJECT_ROOT / "SLAM" / calibration_path

    if args.mock:
        try:
            return 0 if run_camera_preview(args.camera_index) else 1
        except (OSError, RuntimeError, ValueError) as error:
            print("Mission error: {}".format(error))
            return 1

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_root = Path(args.output_dir)
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        output_path = run_dir / "explored_map.json"
    print("Mission output folder: {}".format(output_path.parent))

    try:
        success = run_hardware(args.conn_type, calibration_path, output_path)
        image_path = render_map_image(output_path)
        print("Map image saved to: {}".format(image_path))
        return 0 if success else 1
    except (OSError, RuntimeError, ValueError) as error:
        print("Mission error: {}".format(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
