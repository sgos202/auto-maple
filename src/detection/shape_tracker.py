"""Track a moving, changing silhouette using its outline features."""

import cv2
import numpy as np


def remove_green_cursor(frame):
    """Keep the cursor marker from becoming part of the target silhouette."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 90, 80), (95, 255, 255))
    mask = cv2.dilate(mask, np.ones((11, 11), dtype=np.uint8))
    return cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)


def _gray(frame):
    clean = remove_green_cursor(frame)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _silhouette_mask(frame):
    hsv = cv2.cvtColor(remove_green_cursor(frame), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 170), (180, 40, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def find_initial_shape(frame):
    """Find the bright starting silhouette near the center of the screen."""

    mask = _silhouette_mask(frame)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    height, width = mask.shape
    screen_center = np.array((width / 2, height / 2))
    candidates = []

    for index in range(1, count):
        x, y, box_width, box_height, area = stats[index]
        center = centers[index]
        distance = np.linalg.norm(center - screen_center)
        if area < max(250, int(width * height * 0.001)):
            continue
        if area > width * height * 0.12 or distance > min(width, height) * 0.22:
            continue
        if box_width < 15 or box_height < 15:
            continue
        candidates.append((area / (1 + distance), index))

    if not candidates:
        return None

    _, index = max(candidates)
    x, y, box_width, box_height, _ = stats[index]
    return int(x), int(y), int(box_width), int(box_height)


class ShapeTracker:
    """Estimate a changing silhouette from optical flow along its outline."""

    def __init__(self, frame, box):
        x, y, width, height = box
        self.center = np.array(
            (x + width / 2, y + height / 2), dtype=np.float32
        )
        self.width = float(width)
        self.height = float(height)
        self.previous_gray = _gray(frame)
        self.previous_points = self._find_points(frame, box)
        self.outline = self._make_outline(self.previous_points, box)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.confidence = 1.0
        self.box = box

    @staticmethod
    def _find_points(frame, box):
        x, y, width, height = box
        gray = _gray(frame)
        height_frame, width_frame = gray.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width_frame, x + width), min(height_frame, y + height)
        if x1 <= x0 or y1 <= y0:
            return None

        region = gray[y0:y1, x0:x1]
        binary = _silhouette_mask(frame)[y0:y1, x0:x1]
        count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
        if count <= 1:
            return None
        expected = np.array(((x1 - x0) / 2, (y1 - y0) / 2))
        candidates = [
            i for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= 100
        ]
        if not candidates:
            return None
        index = min(candidates, key=lambda i: np.linalg.norm(centers[i] - expected))
        component = np.uint8(labels == index) * 255
        contour = cv2.Canny(component, 20, 60)
        mask = cv2.dilate(contour, np.ones((3, 3), np.uint8))
        points = cv2.goodFeaturesToTrack(
            region,
            maxCorners=60,
            qualityLevel=0.01,
            minDistance=3,
            mask=mask,
            blockSize=5,
        )
        if points is None:
            return None
        points[:, 0, 0] += x0
        points[:, 0, 1] += y0
        return points

    @staticmethod
    def _make_outline(points, fallback_box):
        if points is None or len(points) < 3:
            x, y, width, height = fallback_box
            return np.array(
                [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                dtype=np.int32,
            )
        return cv2.convexHull(np.round(points[:, 0, :]).astype(np.int32))

    def update(self, frame, elapsed):
        """Track the outline; use recent motion when some edge points are lost."""

        elapsed = max(elapsed, 1 / 120)
        current_gray = _gray(frame)
        height, width = current_gray.shape
        prediction = self.center + self.velocity * elapsed
        tracked = None

        if self.previous_points is not None and len(self.previous_points) >= 3:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                current_gray,
                self.previous_points,
                None,
                winSize=(31, 31),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    20,
                    0.03,
                ),
            )
            if next_points is not None and status is not None:
                good_old = self.previous_points[status.ravel() == 1]
                good_new = next_points[status.ravel() == 1]
                if len(good_new) >= 3:
                    displacements = (good_new - good_old)[:, 0, :]
                    median = np.median(displacements, axis=0)
                    residuals = np.linalg.norm(displacements - median, axis=1)
                    inliers = residuals < max(3, min(self.width, self.height) * 0.12)
                    if np.count_nonzero(inliers) >= 3:
                        displacement = np.median(displacements[inliers], axis=0)
                        tracked = self.center + displacement
                        self.confidence = min(1.0, np.count_nonzero(inliers) / 12)
                    self.previous_points = good_new.reshape(-1, 1, 2)

        if tracked is not None:
            delta = tracked - self.center
            max_step = max(12, min(self.width, self.height) * 0.65)
            distance = float(np.linalg.norm(delta))
            if distance > max_step:
                delta *= max_step / distance
            self.center = 0.65 * (self.center + delta) + 0.35 * prediction
            measured_velocity = delta / elapsed
            self.velocity = 0.65 * self.velocity + 0.35 * measured_velocity
        else:
            self.confidence *= 0.9
            self.center = prediction
            self.velocity *= 0.95

        self.center = np.clip(self.center, (0, 0), (width - 1, height - 1))
        if self.previous_points is not None and len(self.previous_points) >= 3:
            self.outline = self._make_outline(self.previous_points, self.box)
            x, y, box_width, box_height = cv2.boundingRect(self.outline)
            self.box = (x, y, box_width, box_height)
        self.previous_gray = current_gray
        return self.center.copy(), self.confidence, self.box
