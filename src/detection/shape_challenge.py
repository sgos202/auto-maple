"""Experimental live/replay runner for the moving-shape challenge tracker."""

import argparse
import ctypes
import csv
import time
from pathlib import Path

import cv2
import mss
import numpy as np

from src.detection.shape_tracker import ShapeTracker, find_initial_shape


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def _game_window(title):
    user32 = ctypes.windll.user32
    handle = user32.FindWindowW(None, title)
    if not handle:
        raise RuntimeError(f"Could not find a window titled {title!r}.")

    rect = RECT()
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        raise ctypes.WinError()
    if rect.right <= rect.left or rect.bottom <= rect.top:
        raise RuntimeError(f"The {title!r} window has invalid bounds.")
    return handle, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def _move_cursor(position):
    x, y = map(int, position)
    if not ctypes.windll.user32.SetCursorPos(x, y):
        raise ctypes.WinError()


def _cursor_marker(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 100, 100), (95, 255, 255))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 100]
    if not candidates:
        return None
    index = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return centers[index]


def _annotate(
    frame,
    center,
    box,
    confidence,
    cursor=None,
    outline=None,
    predicted=None,
    flow_points=None,
    rejected_candidates=None,
    candidate_outlines=None,
    visible=True,
    reference_outline=None,
):
    output = frame.copy()
    x, y, width, height = box
    box_color = (0, 220, 0) if visible else (0, 220, 255)
    cv2.rectangle(output, (x, y), (x + width, y + height), box_color, 2)
    if outline is not None and visible:
        cv2.polylines(output, [outline], True, (255, 255, 255), 1)
    if reference_outline is not None and not visible:
        cv2.polylines(output, [reference_outline], True, (255, 255, 0), 1)
    if flow_points is not None:
        for point in np.round(flow_points[:, 0, :]).astype(int):
            cv2.circle(output, tuple(point), 2, (255, 0, 0), -1)
    if predicted is not None:
        cv2.circle(
            output, tuple(np.round(predicted).astype(int)), 5, (0, 255, 255), 1
        )
    point = tuple(np.round(center).astype(int))
    if visible:
        cv2.circle(output, point, 4, (0, 220, 0), -1)
    cv2.drawMarker(
        output, point, (255, 255, 255), cv2.MARKER_CROSS, 12, 1
    )
    if cursor is not None:
        cv2.arrowedLine(output, point, tuple(np.round(cursor).astype(int)),
                        (255, 0, 0), 2, tipLength=0.25)
    state = "visible" if visible else "predicting"
    cv2.putText(output, f"{state} {confidence:.2f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return output


def _open_metrics(path):
    if path is None:
        return None, None
    stream = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "frame",
            "time_seconds",
            "confidence",
            "visible",
            "estimate_x",
            "estimate_y",
            "predicted_x",
            "predicted_y",
            "cursor_x",
            "cursor_y",
            "cursor_reference_offset_px",
            "cursor_inside_outline",
        ),
    )
    writer.writeheader()
    return stream, writer


def _write_metrics(writer, frame_index, timestamp, tracker, cursor):
    if writer is None:
        return
    center = tracker.center if tracker is not None else None
    predicted = tracker.predicted if tracker is not None else None
    reference_offset = (
        float(np.linalg.norm(center - cursor))
        if center is not None and cursor is not None
        else ""
    )
    inside_outline = ""
    if tracker is not None and cursor is not None and tracker.visible:
        inside_outline = cv2.pointPolygonTest(
            tracker.outline, tuple(map(float, cursor)), False
        ) >= 0
    writer.writerow(
        {
            "frame": frame_index,
            "time_seconds": f"{timestamp:.4f}",
            "confidence": f"{tracker.confidence:.4f}" if tracker else "",
            "visible": tracker.visible if tracker else "",
            "estimate_x": f"{center[0]:.2f}" if center is not None else "",
            "estimate_y": f"{center[1]:.2f}" if center is not None else "",
            "predicted_x": f"{predicted[0]:.2f}" if predicted is not None else "",
            "predicted_y": f"{predicted[1]:.2f}" if predicted is not None else "",
            "cursor_x": f"{cursor[0]:.2f}" if cursor is not None else "",
            "cursor_y": f"{cursor[1]:.2f}" if cursor is not None else "",
            "cursor_reference_offset_px": (
                f"{reference_offset:.2f}" if reference_offset != "" else ""
            ),
            "cursor_inside_outline": inside_outline,
        }
    )


def run_video(video_path, output_path, metrics_path=None):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if output_path:
        output_path = Path(output_path)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output_path}")

    try:
        metrics_stream, metrics_writer = _open_metrics(metrics_path)
    except OSError:
        capture.release()
        if writer:
            writer.release()
        raise
    tracker = None
    frame_index = 0
    tracked_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if tracker is None:
                box = find_initial_shape(frame)
                if box is None:
                    _write_metrics(
                        metrics_writer, frame_index, frame_index / fps, None, None
                    )
                    if writer:
                        writer.write(frame)
                    frame_index += 1
                    continue
                tracker = ShapeTracker(frame, box)
                center = tracker.center.copy()
                confidence = 1.0
                box = tracker.box
            else:
                center, confidence, box = tracker.update(frame, 1 / fps)

            marker = _cursor_marker(frame)
            _write_metrics(
                metrics_writer, frame_index, frame_index / fps, tracker, marker
            )
            annotated = _annotate(
                frame,
                center,
                box,
                confidence,
                marker,
                tracker.outline,
                tracker.predicted,
                tracker.flow_points,
                tracker.rejected_candidates,
                tracker.candidate_outlines,
                tracker.visible,
                tracker.reference_outline,
            )
            if writer:
                writer.write(annotated)

            tracked_frames += 1
            if frame_index % max(1, round(fps)) == 0:
                print(
                    f"[REPLAY] {frame_index / fps:5.1f}s center="
                    f"({center[0]:.0f}, {center[1]:.0f}) "
                    f"track={confidence:.2f} "
                    f"{'visible' if tracker.visible else 'predicting'}"
                )
            frame_index += 1
    finally:
        capture.release()
        if writer:
            writer.release()
        if metrics_stream:
            metrics_stream.close()

    if not tracked_frames:
        raise RuntimeError("No target was initialized; is its bright countdown shape visible?")
    if output_path:
        print(f"[REPLAY] Annotated video saved to {output_path}")
    if metrics_path:
        print(f"[REPLAY] Per-frame metrics saved to {metrics_path}")


def run_live(title, move_cursor, duration, timeout, metrics_path=None):
    ctypes.windll.user32.SetProcessDPIAware()
    handle, left, top, width, height = _game_window(title)
    region = {"left": left, "top": top, "width": width, "height": height}
    print(
        f"[TRACKER] Watching {title!r} at {region}. "
        "Focus the game to begin; Alt+Tab out to stop."
    )
    if move_cursor:
        print("[TRACKER] Cursor following is ON; no clicks or key presses will be sent.")
    else:
        print("[TRACKER] Preview mode: pass --move-cursor to enable cursor movement.")

    tracker = None
    started_at = time.monotonic()
    previous_time = None
    tracking_started = None
    last_report = 0
    frame_index = 0
    metrics_stream, metrics_writer = _open_metrics(metrics_path)

    try:
        with mss.mss() as screen:
            while True:
                current_handle, current_left, current_top, current_width, current_height = (
                    _game_window(title)
                )
                if current_handle != handle:
                    raise RuntimeError("The game window changed while tracking.")
                if ctypes.windll.user32.GetForegroundWindow() != handle:
                    if tracker is not None:
                        print("[TRACKER] Game focus lost; stopped without sending input.")
                        break
                    if time.monotonic() - started_at > timeout:
                        raise TimeoutError(
                            "The game did not become the foreground window before timeout."
                        )
                    time.sleep(0.05)
                    continue
                left, top = current_left, current_top
                region = {
                    "left": left,
                    "top": top,
                    "width": current_width,
                    "height": current_height,
                }
                frame = np.asarray(screen.grab(region))[:, :, :3]
                now = time.monotonic()
                if tracker is None:
                    box = find_initial_shape(frame)
                    if box is not None:
                        tracker = ShapeTracker(frame, box)
                        tracking_started = now
                        print("[TRACKER] Initial shape found; following its center.")
                    elif now - started_at > timeout:
                        raise TimeoutError(
                            "No bright starting shape found before timeout. "
                            "Start this tool before the challenge fades."
                        )
                else:
                    center, confidence, _ = tracker.update(
                        frame, now - (previous_time or now)
                    )
                    prediction_is_recent = (
                        tracker.lost_frames <= 6 and confidence >= 0.25
                    )
                    if move_cursor and (
                        tracker.visible or prediction_is_recent
                    ):
                        cursor_x = np.clip(center[0], 0, current_width - 1)
                        cursor_y = np.clip(center[1], 0, current_height - 1)
                        _move_cursor((left + cursor_x, top + cursor_y))
                    if now - last_report >= 1:
                        state = "visible" if tracker.visible else "predicting"
                        print(
                            f"[TRACKER] center=({center[0]:.0f}, {center[1]:.0f}) "
                            f"track={confidence:.2f} {state}"
                        )
                        last_report = now
                    if now - tracking_started >= duration:
                        print("[TRACKER] Test duration reached.")
                        break
                marker = _cursor_marker(frame)
                _write_metrics(
                    metrics_writer,
                    frame_index,
                    now - started_at,
                    tracker,
                    marker,
                )
                previous_time = now
                frame_index += 1
                time.sleep(0.01)
    finally:
        if metrics_stream:
            metrics_stream.close()

    print("[TRACKER] Stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Experimental moving-shape tracker for the MapleStory challenge."
    )
    parser.add_argument("--title", default="MapleStory",
                        help="Exact game-window title (default: MapleStory).")
    parser.add_argument("--move-cursor", action="store_true",
                        help="Move the mouse to the estimated shape center.")
    parser.add_argument("--duration", type=float, default=20,
                        help="Maximum live tracking duration after startup.")
    parser.add_argument("--timeout", type=float, default=90,
                        help="Seconds to wait for the bright starting shape.")
    parser.add_argument("--video", help="Replay a recording instead of capturing the game.")
    parser.add_argument("--output", help="Write an annotated replay video to this path.")
    parser.add_argument(
        "--metrics",
        help="Write per-frame confidence, estimate, and visible-cursor measurements to CSV.",
    )
    args = parser.parse_args()

    if args.video:
        if args.move_cursor:
            parser.error("--move-cursor is only available in live mode.")
        run_video(args.video, args.output, args.metrics)
        return
    if args.output:
        parser.error("--output is only available with --video.")
    if args.duration <= 0 or args.timeout <= 0:
        parser.error("--duration and --timeout must be positive.")
    run_live(args.title, args.move_cursor, args.duration, args.timeout, args.metrics)


if __name__ == "__main__":
    main()
