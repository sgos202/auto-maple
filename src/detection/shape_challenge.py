"""Experimental live/replay runner for the moving-shape challenge tracker."""

import argparse
import ctypes
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


def _annotate(frame, center, box, confidence, cursor=None, outline=None):
    output = frame.copy()
    x, y, width, height = box
    cv2.rectangle(output, (x, y), (x + width, y + height), (0, 220, 0), 2)
    point = tuple(np.round(center).astype(int))
    if outline is not None:
        cv2.polylines(output, [outline], True, (0, 220, 0), 2)
    cv2.circle(output, point, 4, (0, 0, 255), -1)
    if cursor is not None:
        cv2.arrowedLine(output, point, tuple(np.round(cursor).astype(int)),
                        (255, 0, 0), 2, tipLength=0.25)
    cv2.putText(output, f"track {confidence:.2f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return output


def run_video(video_path, output_path):
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

    tracker = None
    inside_outline = []
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if tracker is None:
                box = find_initial_shape(frame)
                if box is None:
                    continue
                tracker = ShapeTracker(frame, box)
                center = tracker.center.copy()
                confidence = 1.0
                box = tracker.box
            else:
                center, confidence, box = tracker.update(frame, 1 / fps)

            marker = _cursor_marker(frame)
            if marker is not None:
                inside_outline.append(
                    cv2.pointPolygonTest(
                        tracker.outline, tuple(map(float, marker)), False
                    ) >= 0
                )
            annotated = _annotate(
                frame, center, box, confidence, marker, tracker.outline
            )
            if writer:
                writer.write(annotated)

            frames += 1
            if frames % max(1, round(fps)) == 0:
                print(
                    f"[REPLAY] {frames / fps:5.1f}s center="
                    f"({center[0]:.0f}, {center[1]:.0f}) "
                    f"track={confidence:.2f}"
                )
    finally:
        capture.release()
        if writer:
            writer.release()

    if not frames:
        raise RuntimeError("No target was initialized; is its bright countdown shape visible?")
    if inside_outline:
        print(
            "[REPLAY] Recorded cursor inside estimated outline: "
            f"{100 * np.mean(inside_outline):.1f}% of frames."
        )
    if output_path:
        print(f"[REPLAY] Annotated video saved to {output_path}")


def run_live(title, move_cursor, duration, timeout):
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
                if move_cursor:
                    _move_cursor((left + center[0], top + center[1]))
                if now - last_report >= 1:
                    print(
                        f"[TRACKER] center=({center[0]:.0f}, {center[1]:.0f}) "
                        f"track={confidence:.2f}"
                    )
                    last_report = now
                if now - tracking_started >= duration:
                    print("[TRACKER] Test duration reached.")
                    break
            previous_time = now
            time.sleep(0.01)

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
    args = parser.parse_args()

    if args.video:
        if args.move_cursor:
            parser.error("--move-cursor is only available in live mode.")
        run_video(args.video, args.output)
        return
    if args.output:
        parser.error("--output is only available with --video.")
    if args.duration <= 0 or args.timeout <= 0:
        parser.error("--duration and --timeout must be positive.")
    run_live(args.title, args.move_cursor, args.duration, args.timeout)


if __name__ == "__main__":
    main()
