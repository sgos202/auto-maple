"""Track the make-up Violetta and optionally select its answer in the mini-game."""

import argparse
import ctypes
import time

import cv2
import mss
import numpy as np


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class ViolettaTracker:
    """Track the make-up Violetta identified during the opening reveal."""

    INITIAL_Y = 0.30
    INITIAL_WIDTH = 0.173
    INITIAL_HEIGHT = 0.30
    SLOT_CENTERS = (0.195, 0.406, 0.618, 0.829)
    MIN_FEATURES = 8
    GOOD_FEATURES = 28
    MIN_LINEUP_PURPLE_COVERAGE = 0.25
    MIN_MAKEUP_COVERAGE = 0.02
    MIN_MAKEUP_SCORE_MARGIN = 0.015

    def __init__(self, frame):
        height, width = frame.shape[:2]
        slot_index = self.find_makeup_slot(frame)
        if slot_index is None:
            raise ValueError(
                "Could not confidently identify the make-up Violetta. "
                "Start tracking while the four Violettas are shown in the "
                "opening reveal."
            )

        box_width = round(width * self.INITIAL_WIDTH)
        box_height = round(height * self.INITIAL_HEIGHT)
        x = round(width * self.SLOT_CENTERS[slot_index] - box_width / 2)
        y = round(height * self.INITIAL_Y)
        self._initialize_box(frame, (x, y, x + box_width, y + box_height))

    @classmethod
    def from_box(cls, frame, box):
        tracker = cls.__new__(cls)
        tracker._initialize_box(frame, box)
        return tracker

    def _initialize_box(self, frame, box):
        if len(box) != 4:
            raise ValueError("A tracking box must contain left, top, right, and bottom.")
        x0, y0, x1, y1 = map(float, box)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("A tracking box must have positive width and height.")
        self.width = x1 - x0
        self.height = y1 - y0
        self.center = np.array(((x0 + x1) / 2, (y0 + y1) / 2), dtype=np.float32)
        self._previous_gray = self._gray(frame)
        self._points = self._find_points(self._previous_gray, frame)
        if self._points is None or len(self._points) < self.MIN_FEATURES:
            raise ValueError(
                "Could not find enough visual features in the target box."
            )
        self.confidence = 1.0
        self.visible = True
        self.lost_frames = 0
        self._lane_history = []

    @classmethod
    def find_makeup_slot(cls, frame):
        """Find the unique blue-eye make-up among the four opening characters."""

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HSV)
        purple_coverages = []
        makeup_coverages = []
        for center_x in cls.SLOT_CENTERS:
            head = hsv[
                round(height * 0.38):round(height * 0.54),
                round(width * (center_x - 0.08)):round(width * (center_x + 0.08)),
            ]
            purple = cv2.inRange(head, (135, 50, 50), (180, 255, 255))
            purple_coverages.append(
                cv2.countNonZero(purple) / max(purple.size, 1)
            )

            eyes = hsv[
                round(height * 0.515):round(height * 0.575),
                round(width * (center_x - 0.035)):round(width * (center_x + 0.035)),
            ]
            blue = cv2.inRange(eyes, (85, 90, 50), (135, 255, 255))
            makeup_coverages.append(
                cv2.countNonZero(blue) / max(blue.size, 1)
            )

        if min(purple_coverages) < cls.MIN_LINEUP_PURPLE_COVERAGE:
            return None

        ranking = sorted(
            enumerate(makeup_coverages),
            key=lambda item: item[1],
            reverse=True,
        )
        (slot_index, best_score), (_, second_score) = ranking[:2]
        if (
            best_score < cls.MIN_MAKEUP_COVERAGE
            or best_score - second_score < cls.MIN_MAKEUP_SCORE_MARGIN
        ):
            return None
        return slot_index

    @staticmethod
    def _gray(frame):
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)

    def _box(self):
        x = round(float(self.center[0] - self.width / 2))
        y = round(float(self.center[1] - self.height / 2))
        return x, y, round(self.width), round(self.height)

    def _point_mask(self, shape, center=None, width=None, height=None):
        if center is None:
            center = self.center
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        frame_height, frame_width = shape
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        x, y = center
        polygon = np.array(
            [
                (x, y - height * 0.48),
                (x + width * 0.34, y - height * 0.36),
                (x + width * 0.47, y - height * 0.02),
                (x + width * 0.34, y + height * 0.30),
                (x + width * 0.23, y + height * 0.48),
                (x, y + height * 0.50),
                (x - width * 0.23, y + height * 0.48),
                (x - width * 0.34, y + height * 0.30),
                (x - width * 0.47, y - height * 0.02),
                (x - width * 0.34, y - height * 0.36),
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def _find_points(self, gray, frame):
        x, y, width, height = self._box()
        frame_height, frame_width = gray.shape
        x0, y0 = max(0, x), max(0, y)
        x1 = min(frame_width, x + width)
        y1 = min(frame_height, y + height)
        if x1 <= x0 or y1 <= y0:
            return None

        mask = self._point_mask(gray.shape)
        hsv = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HSV)
        mask[hsv[:, :, 1] < 25] = 0
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,
            qualityLevel=0.015,
            minDistance=4,
            mask=mask,
            blockSize=5,
        )
        return points

    def _points_cover_character(self, points, center=None, width=None, height=None):
        if center is None:
            center = self.center
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        xy = points.reshape(-1, 2)
        return (
            np.ptp(xy[:, 0]) >= width * 0.12
            and np.ptp(xy[:, 1]) >= height * 0.12
        )

    def update(self, frame):
        if not self.visible:
            self._previous_gray = self._gray(frame)
            self.lost_frames += 1
            self.confidence = 0.0
            return

        gray = self._gray(frame)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            self._points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )
        if next_points is None or status is None:
            self._mark_lost(gray)
            return

        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self._previous_gray,
            next_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )
        if backward_points is None or backward_status is None:
            self._mark_lost(gray)
            return

        forward_backward_error = np.linalg.norm(
            self._points[:, 0] - backward_points[:, 0], axis=1
        )
        valid = (
            (status.ravel() == 1)
            & (backward_status.ravel() == 1)
            & (forward_backward_error < 2.5)
        )
        old_points = self._points[valid]
        new_points = next_points[valid]
        affine, inliers = (None, None)
        if len(new_points) >= self.MIN_FEATURES:
            affine, inliers = cv2.estimateAffinePartial2D(
                old_points,
                new_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
            )

        if affine is None or inliers is None:
            self._mark_lost(gray)
            return

        inlier_mask = inliers.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inlier_count / len(new_points)
        scale = float(np.hypot(affine[0, 0], affine[1, 0]))
        if (
            inlier_count < self.MIN_FEATURES
            or inlier_ratio < 0.3
            or not 0.7 <= scale <= 1.4
            or not self._points_cover_character(old_points[inlier_mask])
        ):
            self._mark_lost(gray)
            return

        transformed_center = affine @ np.array(
            (self.center[0], self.center[1], 1), dtype=np.float32
        )
        displacement = float(np.linalg.norm(transformed_center - self.center))
        maximum_displacement = max(60.0, min(gray.shape) * 0.2)
        if displacement > maximum_displacement:
            self._mark_lost(gray)
            return

        transformed_points = new_points[inlier_mask]
        character_mask = self._point_mask(
            gray.shape,
            transformed_center,
            self.width * scale,
            self.height * scale,
        )
        rounded_points = np.rint(transformed_points[:, 0]).astype(int)
        inside = (
            (rounded_points[:, 0] >= 0)
            & (rounded_points[:, 0] < gray.shape[1])
            & (rounded_points[:, 1] >= 0)
            & (rounded_points[:, 1] < gray.shape[0])
        )
        in_character = np.zeros(len(inside), dtype=bool)
        in_character[inside] = (
            character_mask[
                rounded_points[inside, 1], rounded_points[inside, 0]
            ] > 0
        )
        if np.count_nonzero(in_character) < self.MIN_FEATURES:
            self._mark_lost(gray)
            return
        transformed_points = transformed_points[in_character]
        if not self._points_cover_character(
            transformed_points,
            transformed_center,
            self.width * scale,
            self.height * scale,
        ):
            self._mark_lost(gray)
            return

        self.center = transformed_center.astype(np.float32)
        self.width *= scale
        self.height *= scale
        self._points = transformed_points.reshape(-1, 1, 2)
        self._previous_gray = gray
        self.visible = True
        self.lost_frames = 0
        self.confidence = min(1.0, inlier_count / self.GOOD_FEATURES) * inlier_ratio
        self._refresh_features(gray, frame)
        self._lane_history.append(self.answer_index(gray.shape[1]))
        self._lane_history = self._lane_history[-8:]

    def _mark_lost(self, gray):
        self._previous_gray = gray
        self._points = np.empty((0, 1, 2), dtype=np.float32)
        self.visible = False
        self.lost_frames += 1
        self.confidence = 0.0
        self._lane_history.clear()

    def _refresh_features(self, gray, frame):
        if len(self._points) >= self.GOOD_FEATURES:
            return
        new_points = self._find_points(gray, frame)
        if new_points is None:
            return
        if len(self._points):
            distances = np.linalg.norm(
                new_points[:, 0, None, :] - self._points[None, :, 0, :],
                axis=2,
            )
            new_points = new_points[np.all(distances > 4, axis=1)]
        if len(new_points):
            self._points = np.concatenate((self._points, new_points), axis=0)[:100]

    def answer_index(self, frame_width):
        return min(
            range(len(self.SLOT_CENTERS)),
            key=lambda index: abs(
                self.center[0] / frame_width - self.SLOT_CENTERS[index]
            ),
        )

    def lane_is_stable(self):
        return (
            len(self._lane_history) >= 5
            and len(set(self._lane_history[-5:])) == 1
        )

    def draw(self, frame):
        output = frame.copy()
        x, y, width, height = self._box()
        color = (0, 220, 0) if self.visible else (0, 180, 255)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        label = (
            f"Violetta {self.answer_index(frame.shape[1]) + 1} "
            f"{'tracked' if self.visible else 'lost'} {self.confidence:.2f}"
        )
        cv2.putText(
            output, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )
        return output


def _countdown_confidence(frame):
    """Return red-digit coverage in the timer area near the top-center."""

    height, width = frame.shape[:2]
    x1, x2 = round(width * 0.46), round(width * 0.55)
    y1, y2 = round(height * 0.19), round(height * 0.36)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 120, 120), (12, 255, 255))
    red |= cv2.inRange(hsv, (168, 120, 120), (180, 255, 255))
    region = red[y1:y2, x1:x2]
    return cv2.countNonZero(region) / max(region.size, 1)


def _client_region(title):
    user32 = ctypes.windll.user32
    handle = user32.FindWindowW(None, title)
    if not handle:
        raise RuntimeError(f"Could not find a window titled {title!r}.")

    rect = RECT()
    if not user32.GetClientRect(handle, ctypes.byref(rect)):
        raise ctypes.WinError()
    origin = POINT(0, 0)
    if not user32.ClientToScreen(handle, ctypes.byref(origin)):
        raise ctypes.WinError()
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        raise RuntimeError(f"The {title!r} window has invalid client bounds.")
    return handle, {
        "left": origin.x,
        "top": origin.y,
        "width": width,
        "height": height,
    }


def _click_answer(region, answer_index):
    user32 = ctypes.windll.user32
    x = region["left"] + round(
        region["width"] * ViolettaTracker.SLOT_CENTERS[answer_index]
    )
    y = region["top"] + round(region["height"] * 0.845)
    from src.common import vkeys
    vkeys.move_cursor(x, y, strict=True)
    vkeys.click(None, allow_paused=True)


def _video(video_path, output_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if output_path:
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
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if tracker is None:
                tracker = ViolettaTracker(frame)
                print(
                    "[REPLAY] Identified the make-up Violetta in answer "
                    f"{tracker.answer_index(width) + 1}."
                )
            else:
                tracker.update(frame)
            if writer:
                writer.write(tracker.draw(frame))
            frame_index += 1
            if frame_index % max(1, round(fps)) == 0:
                print(
                    f"[REPLAY] {frame_index / fps:.1f}s "
                    f"answer={tracker.answer_index(width) + 1} "
                    f"confidence={tracker.confidence:.2f} "
                    f"{'tracked' if tracker.visible else 'lost'}"
                )
    finally:
        capture.release()
        if writer:
            writer.release()

    if tracker is None:
        raise RuntimeError("The supplied video contained no frames.")
    print(
        f"[REPLAY] Final tracked answer: {tracker.answer_index(width) + 1}; "
        f"{'tracked' if tracker.visible else 'lost'}."
    )
    if output_path:
        print(f"[REPLAY] Annotated video saved to {output_path}")


def _live(title, timeout, reveal_timeout, select):
    ctypes.windll.user32.SetProcessDPIAware()
    handle, region = _client_region(title)
    tracker = None
    started = time.monotonic()
    last_report = 0.0

    print(
        f"[TRACKER] Watching {title!r}. Start on the mini-game's initial screen "
        "with the four Violettas visible."
    )
    if select:
        print("[TRACKER] Automatic answer selection is ON.")
    else:
        print("[TRACKER] Preview only; pass --select to click the answer.")

    reveal_started = time.monotonic()
    with mss.MSS() as screen:
        while time.monotonic() - started < timeout:
            current_handle, region = _client_region(title)
            if current_handle != handle:
                raise RuntimeError("The game window changed while tracking.")
            if ctypes.windll.user32.GetForegroundWindow() != handle:
                if tracker is not None:
                    raise RuntimeError(
                        "Game focus was lost; stopped without selecting an answer."
                    )
                time.sleep(0.05)
                continue

            frame = np.asarray(screen.grab(region))[:, :, :3]
            if tracker is None:
                try:
                    tracker = ViolettaTracker(frame)
                except ValueError:
                    if time.monotonic() - reveal_started >= reveal_timeout:
                        raise TimeoutError(
                            "Could not identify the make-up Violetta during the "
                            "opening reveal. Restart before the four characters "
                            "begin moving."
                        )
                    time.sleep(0.03)
                    continue
                print(
                    "[TRACKER] Identified the make-up Violetta in answer "
                    f"{tracker.answer_index(frame.shape[1]) + 1}."
                )
            else:
                tracker.update(frame)

            now = time.monotonic()
            if now - last_report >= 1:
                state = "tracked" if tracker.visible else "lost"
                print(
                    f"[TRACKER] answer={tracker.answer_index(frame.shape[1]) + 1} "
                    f"confidence={tracker.confidence:.2f} {state}"
                )
                last_report = now

            if _countdown_confidence(frame) > 0.10:
                if not select:
                    print(
                        f"[TRACKER] Answer countdown detected. "
                        f"Target is button {tracker.answer_index(frame.shape[1]) + 1}."
                    )
                    return
                if (
                    tracker.visible
                    and tracker.confidence >= 0.18
                    and tracker.lane_is_stable()
                ):
                    answer = tracker.answer_index(frame.shape[1])
                    _click_answer(region, answer)
                    print(
                        f"[TRACKER] Selected answer {answer + 1} "
                        f"at confidence {tracker.confidence:.2f}."
                    )
                    return
                print(
                    "[TRACKER] Countdown detected, but the target track is not "
                    "confident and stable; no answer was selected."
                )
                return

            time.sleep(0.01)

    if tracker is None:
        raise TimeoutError(
            "The mini-game was not detected before timeout. Start tracking with "
            "the four Violettas visible."
        )
    raise TimeoutError(
        "No answer countdown appeared before timeout; no answer was selected."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Identify and track the make-up Violetta from its random starting position."
    )
    parser.add_argument("--title", default="MapleStory",
                        help="Exact game-window title (default: MapleStory).")
    parser.add_argument("--timeout", type=float, default=90,
                        help="Seconds to wait for the answer countdown.")
    parser.add_argument(
        "--reveal-timeout",
        type=float,
        default=3,
        help="Seconds allowed to identify the target during the opening reveal.",
    )
    parser.add_argument("--select", action="store_true",
                        help="Click the tracked answer when the red countdown appears.")
    parser.add_argument("--video", help="Replay a recording instead of live capture.")
    parser.add_argument("--output", help="Write an annotated replay video.")
    args = parser.parse_args()

    if args.timeout <= 0 or args.reveal_timeout <= 0:
        parser.error("--timeout and --reveal-timeout must be positive.")
    if args.video:
        if args.select:
            parser.error("--select is available only in live mode.")
        _video(args.video, args.output)
        return
    if args.output:
        parser.error("--output is available only with --video.")
    _live(args.title, args.timeout, args.reveal_timeout, args.select)


if __name__ == "__main__":
    main()
