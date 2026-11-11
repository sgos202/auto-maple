"""Track a changing silhouette using optical flow and confidence-gated matching."""

import cv2
import numpy as np


def remove_green_cursor(frame):
    """Keep the cursor marker from becoming part of the target silhouette."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 90, 80), (95, 255, 255))
    mask = cv2.dilate(mask, np.ones((11, 11), dtype=np.uint8))
    return cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)


def _gray(frame, clean_frame=None):
    clean = clean_frame if clean_frame is not None else remove_green_cursor(frame)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _silhouette_mask(frame, clean_frame=None):
    clean = clean_frame if clean_frame is not None else remove_green_cursor(frame)
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 170), (180, 40, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _largest_contour(mask, offset=(0, 0)):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).copy()
    contour[:, 0, 0] += offset[0]
    contour[:, 0, 1] += offset[1]
    return contour


def _aim_point(component, x, y):
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    maximum, _, _, location = cv2.minMaxLoc(distance)
    if (
        maximum < 2
        or location[0] <= 1
        or location[1] <= 1
        or location[0] >= component.shape[1] - 2
        or location[1] >= component.shape[0] - 2
    ):
        return np.array(
            (x + component.shape[1] / 2, y + component.shape[0] / 2),
            dtype=np.float32,
        )
    return np.array((x + location[0], y + location[1]), dtype=np.float32)


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
    """Follow a target outline and predict its position when it is not visible."""

    _MATCH_THRESHOLD = 0.56

    def __init__(self, frame, box):
        x, y, width, height = box
        frame_height, frame_width = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame_width, x + width), min(frame_height, y + height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("The initial target box must overlap the frame.")

        self.box = (x0, y0, x1 - x0, y1 - y0)
        clean_frame = remove_green_cursor(frame)
        initial_mask = _silhouette_mask(frame, clean_frame)[y0:y1, x0:x1]
        self.center = _aim_point(initial_mask, x0, y0)
        if not cv2.countNonZero(initial_mask):
            self.center = np.array(
                (x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2),
                dtype=np.float32,
            )
        self.width = float(x1 - x0)
        self.height = float(y1 - y0)
        self.initial_center = self.center.copy()
        self.template_aim_offset = self.center - np.array(
            (x0, y0), dtype=np.float32
        )
        self.template_mask = initial_mask.copy()
        self.template_area = max(
            float(cv2.countNonZero(initial_mask)),
            self.width * self.height * 0.35,
        )
        self.template_aspect = self.width / max(self.height, 1)
        template_contour = _largest_contour(initial_mask, (x0, y0))
        if template_contour is None:
            template_contour = np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
        perimeter = cv2.arcLength(template_contour, True)
        self.template_contour = cv2.approxPolyDP(
            template_contour, 0.02 * perimeter, True
        )
        self.velocity = np.zeros(2, dtype=np.float32)
        self.previous_gray = _gray(frame, clean_frame)
        edge_mask = np.zeros_like(initial_mask)
        local_contour = self.template_contour.copy()
        local_contour[:, 0, 0] -= x0
        local_contour[:, 0, 1] -= y0
        cv2.drawContours(edge_mask, [local_contour], -1, 255, 5)
        self.previous_points = self._find_points(
            frame, self.box, clean_frame, edge_mask
        )
        self.outline = self.template_contour.copy()
        self.confidence = 1.0
        self.lost_frames = 0
        self.visible = True
        self.predicted = self.center.copy()
        self.reference_outline = self.template_contour.copy()
        self.flow_points = self.previous_points
        self.rejected_candidates = []
        self.candidate_outlines = []
        self.match_scores = {"M": 1.0, "P": 1.0, "O": 1.0, "C": 1.0}

    @staticmethod
    def _find_points(frame, box, clean_frame=None, target_mask=None):
        x, y, width, height = box
        if clean_frame is None:
            clean_frame = remove_green_cursor(frame)
        gray = _gray(frame, clean_frame)
        frame_height, frame_width = gray.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame_width, x + width), min(frame_height, y + height)
        if x1 <= x0 or y1 <= y0:
            return None

        region = gray[y0:y1, x0:x1]
        if target_mask is None:
            target_mask = _silhouette_mask(frame, clean_frame)[y0:y1, x0:x1]
        if target_mask.shape != (y1 - y0, x1 - x0):
            target_mask = cv2.resize(
                target_mask,
                (x1 - x0, y1 - y0),
                interpolation=cv2.INTER_NEAREST,
            )
        feature_mask = cv2.dilate(
            target_mask, np.ones((3, 3), dtype=np.uint8)
        )
        points = cv2.goodFeaturesToTrack(
            region,
            maxCorners=60,
            qualityLevel=0.01,
            minDistance=3,
            mask=feature_mask,
            blockSize=5,
        )
        if points is None or len(points) < 3:
            return None
        points[:, 0, 0] += x0
        points[:, 0, 1] += y0
        return points

    def _flow_displacement(self, current_gray):
        if self.previous_points is None or len(self.previous_points) < 3:
            return None, None, 0.0, None

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
        if next_points is None or status is None:
            return None, None, 0.0, None

        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self.previous_gray,
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
            return None, None, 0.0, None

        round_trip_error = np.linalg.norm(
            backward_points[:, 0, :] - self.previous_points[:, 0, :], axis=1
        )
        valid = (
            (status.ravel() == 1)
            & (backward_status.ravel() == 1)
            & (round_trip_error <= 1.5)
        )
        good_old = self.previous_points[valid]
        good_new = next_points[valid]
        if len(good_new) < 3:
            return None, good_new, 0.0, None

        transform, inlier_mask = cv2.estimateAffinePartial2D(
            good_old,
            good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=100,
            confidence=0.95,
        )
        if transform is None or inlier_mask is None:
            return None, good_new, 0.0, None

        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 3:
            return None, good_new[inliers], inlier_count / len(good_new), None

        scale = float(np.hypot(transform[0, 0], transform[1, 0]))
        rotation = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
        if not 0.90 <= scale <= 1.10 or abs(rotation) > 25:
            return None, good_new[inliers], inlier_count / len(good_new), None

        transformed_center = cv2.transform(
            self.center.reshape(1, 1, 2), transform
        )[0, 0]
        displacement = transformed_center - self.center
        if np.linalg.norm(displacement) > max(
            20, min(self.width, self.height) * 0.65
        ):
            return None, good_new[inliers], 0.0, None
        angle = np.radians(rotation)
        cosine, sine = np.cos(angle), np.sin(angle)
        rigid_transform = np.array(
            [
                [
                    cosine,
                    -sine,
                    transformed_center[0]
                    - cosine * self.center[0]
                    + sine * self.center[1],
                ],
                [
                    sine,
                    cosine,
                    transformed_center[1]
                    - sine * self.center[0]
                    - cosine * self.center[1],
                ],
            ],
            dtype=np.float32,
        )
        return displacement.astype(np.float32), good_new[inliers], (
            inlier_count / max(len(self.previous_points), 1)
        ), rigid_transform

    def _edge_feature_mask(self):
        x, y, width, height = self.box
        mask = np.zeros((height, width), dtype=np.uint8)
        contour = self.outline.copy()
        contour[:, 0, 0] -= x
        contour[:, 0, 1] -= y
        cv2.drawContours(mask, [contour], -1, 255, 5)
        return mask

    def _translate_model(self, displacement):
        dx, dy = np.asarray(displacement, dtype=np.float32).reshape(2)
        transform = np.array(
            ((1.0, 0.0, dx), (0.0, 1.0, dy)), dtype=np.float32
        )
        self.outline = cv2.transform(self.outline, transform)
        x, y, width, height = cv2.boundingRect(self.outline)
        padding = 5
        self.box = (
            x - padding,
            y - padding,
            width + 2 * padding,
            height + 2 * padding,
        )

    def _candidates(self, frame, prediction, search_radius, clean_frame):
        hsv = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = hsv.shape[:2]
        x0 = max(0, int(prediction[0] - search_radius))
        y0 = max(0, int(prediction[1] - search_radius))
        x1 = min(frame_width, int(prediction[0] + search_radius + 1))
        y1 = min(frame_height, int(prediction[1] + search_radius + 1))
        if x1 <= x0 or y1 <= y0:
            return []

        expected_area = self.template_area
        min_area = max(20, int(expected_area * 0.12))
        max_area = max(min_area + 1, int(expected_area * 3.5))
        min_width = max(5, int(self.width * 0.35))
        min_height = max(5, int(self.height * 0.35))
        max_width = max(min_width + 1, int(self.width * 2.8))
        max_height = max(min_height + 1, int(self.height * 2.8))
        candidates = []

        def add_candidate(
            contour, component, candidate_x, candidate_y, edge_only
        ):
            height, width = component.shape
            area = float(cv2.contourArea(contour))
            if area < max(10, expected_area * 0.08):
                return
            if not min_width <= width <= max_width:
                return
            if not min_height <= height <= max_height:
                return
            if edge_only:
                polygon = cv2.approxPolyDP(
                    contour, 0.04 * cv2.arcLength(contour, True), True
                )
                if not 3 <= len(polygon) <= 5 or not cv2.isContourConvex(polygon):
                    return

            moments = cv2.moments(contour)
            if moments["m00"]:
                center = np.array(
                    (
                        moments["m10"] / moments["m00"],
                        moments["m01"] / moments["m00"],
                    ),
                    dtype=np.float32,
                )
            else:
                center = np.array(
                    (candidate_x + width / 2, candidate_y + height / 2),
                    dtype=np.float32,
                )
            if np.linalg.norm(center - prediction) > search_radius:
                return

            filled = np.zeros((height, width), dtype=np.uint8)
            local_contour = contour.copy()
            local_contour[:, 0, 0] -= candidate_x
            local_contour[:, 0, 1] -= candidate_y
            cv2.drawContours(filled, [local_contour], -1, 255, cv2.FILLED)
            aim = _aim_point(filled, candidate_x, candidate_y)
            candidate_hsv = hsv[
                candidate_y:candidate_y + height,
                candidate_x:candidate_x + width,
            ]
            foreground = filled > 0
            mean_value = float(np.mean(candidate_hsv[:, :, 2][foreground]))
            mean_saturation = float(np.mean(candidate_hsv[:, :, 1][foreground]))
            aspect = width / max(height, 1)
            shape_distance = cv2.matchShapes(
                self.template_contour,
                contour,
                cv2.CONTOURS_MATCH_I1,
                0.0,
            )
            if shape_distance > 0.45:
                return
            shape_similarity = float(np.exp(-3.0 * shape_distance))
            area_similarity = float(
                np.exp(-abs(np.log(max(area, 1) / expected_area)))
            )
            aspect_similarity = float(
                np.exp(-abs(np.log(max(aspect, 0.01) / self.template_aspect)))
            )
            if edge_only:
                outline_similarity = (
                    0.80 * shape_similarity + 0.20 * aspect_similarity
                )
            else:
                outline_similarity = (
                    0.65 * shape_similarity
                    + 0.20 * area_similarity
                    + 0.15 * aspect_similarity
                )
            brightness_similarity = float(np.exp(-abs(mean_value - 255) / 140))
            color_similarity = (
                0.65 * brightness_similarity
                + 0.35 * (1.0 - mean_saturation / 255.0)
            )
            candidates.append(
                {
                    "box": (
                        candidate_x,
                        candidate_y,
                        int(width),
                        int(height),
                    ),
                    "center": center,
                    "aim": aim,
                    "area": area,
                    "contour": contour,
                    "component": filled,
                    "outline": outline_similarity,
                    "color": color_similarity,
                    "edge_only": edge_only,
                }
            )

        for max_saturation, min_value in ((40, 170), (65, 145), (90, 125)):
            mask = cv2.inRange(
                hsv[y0:y1, x0:x1],
                (0, 0, min_value),
                (180, max_saturation, 255),
            )
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
            )
            count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)

            for index in range(1, count):
                local_x, local_y, width, height, area = stats[index]
                if not min_area <= area <= max_area:
                    continue
                if not min_width <= width <= max_width:
                    continue
                if not min_height <= height <= max_height:
                    continue

                component = np.uint8(labels == index) * 255
                component_crop = component[
                    local_y:local_y + height, local_x:local_x + width
                ]
                contour = _largest_contour(
                    component_crop, (x0 + local_x, y0 + local_y)
                )
                if contour is None or cv2.contourArea(contour) < 10:
                    continue
                add_candidate(
                    contour,
                    component_crop,
                    x0 + local_x,
                    y0 + local_y,
                    False,
                )

        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 25, 75)
        for kernel_size in (3, 5, 7):
            closed = cv2.morphologyEx(
                edges,
                cv2.MORPH_CLOSE,
                np.ones((kernel_size, kernel_size), dtype=np.uint8),
            )
            contours, _ = cv2.findContours(
                closed[y0:y1, x0:x1],
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for local_contour in contours:
                contour_area = cv2.contourArea(local_contour)
                if contour_area < max(10, expected_area * 0.08):
                    continue
                local_box = cv2.boundingRect(local_contour)
                local_x, local_y, width, height = local_box
                if not min_width <= width <= max_width:
                    continue
                if not min_height <= height <= max_height:
                    continue

                contour = local_contour.copy()
                contour[:, 0, 0] += x0
                contour[:, 0, 1] += y0
                polygon = cv2.approxPolyDP(
                    contour, 0.04 * cv2.arcLength(contour, True), True
                )
                if not 3 <= len(polygon) <= 5 or not cv2.isContourConvex(polygon):
                    continue
                component = np.full(
                    (height, width), 255, dtype=np.uint8
                )
                add_candidate(
                    contour,
                    component,
                    x0 + local_x,
                    y0 + local_y,
                    True,
                )

        unique = []
        duplicate_radius = max(6, min(self.width, self.height) * 0.2)
        candidates.sort(
            key=lambda item: (
                item["outline"],
                not item["edge_only"],
            ),
            reverse=True,
        )
        for candidate in candidates:
            if any(
                np.linalg.norm(candidate["center"] - other["center"])
                < duplicate_radius
                and abs(candidate["box"][2] - other["box"][2])
                < duplicate_radius
                and abs(candidate["box"][3] - other["box"][3])
                < duplicate_radius
                for other in unique
            ):
                continue
            unique.append(candidate)
        return unique

    def _outline_matches(self, frame, prediction, search_radius, clean_frame):
        """Match the saved silhouette boundary against scene edges near prediction."""

        gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 25, 75)
        distance = cv2.distanceTransform(
            cv2.bitwise_not(edges), cv2.DIST_L2, 3
        )
        frame_height, frame_width = gray.shape
        matches = []

        for scale in (0.9, 1.0, 1.1):
            width = max(5, int(round(self.width * scale)))
            height = max(5, int(round(self.height * scale)))
            mask = cv2.resize(
                self.template_mask, (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            contour = _largest_contour(mask)
            if contour is None:
                continue
            outline = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(outline, [contour], -1, 1, 1)
            point_count = int(np.count_nonzero(outline))
            if not point_count:
                continue

            aim_offset = self.template_aim_offset * scale
            left = max(0, int(prediction[0] - search_radius - aim_offset[0]))
            top = max(0, int(prediction[1] - search_radius - aim_offset[1]))
            right = min(
                frame_width,
                int(prediction[0] + search_radius - aim_offset[0] + width + 1),
            )
            bottom = min(
                frame_height,
                int(prediction[1] + search_radius - aim_offset[1] + height + 1),
            )
            if right - left < width or bottom - top < height:
                continue

            response = cv2.matchTemplate(
                distance[top:bottom, left:right].astype(np.float32),
                outline.astype(np.float32),
                cv2.TM_CCORR,
            ) / point_count
            for _ in range(3):
                mean_edge_distance, _, location, _ = cv2.minMaxLoc(response)
                candidate_x, candidate_y = left + location[0], top + location[1]
                aim = np.array(
                    (candidate_x + aim_offset[0], candidate_y + aim_offset[1]),
                    dtype=np.float32,
                )
                if np.linalg.norm(aim - prediction) <= search_radius:
                    candidate_contour = contour.copy()
                    candidate_contour[:, 0, 0] += candidate_x
                    candidate_contour[:, 0, 1] += candidate_y
                    matches.append(
                        {
                            "box": (candidate_x, candidate_y, width, height),
                            "center": aim,
                            "aim": aim,
                            "area": float(cv2.countNonZero(mask)),
                            "contour": candidate_contour,
                            "component": mask,
                            "outline": float(np.exp(-mean_edge_distance / 5.0)),
                            "color": 0.0,
                            "edge_only": True,
                        }
                    )

                x0 = max(0, location[0] - max(3, width // 2))
                y0 = max(0, location[1] - max(3, height // 2))
                x1 = min(response.shape[1], location[0] + max(3, width // 2) + 1)
                y1 = min(response.shape[0], location[1] + max(3, height // 2) + 1)
                response[y0:y1, x0:x1] = np.inf

        return matches

    def update(self, frame, elapsed):
        """Track a visible candidate or predict without switching on weak evidence."""

        elapsed = max(float(elapsed), 1 / 120)
        clean_frame = remove_green_cursor(frame)
        current_gray = _gray(frame, clean_frame)
        frame_height, frame_width = current_gray.shape
        previous_center = self.center.copy()
        prediction = self.center + self.velocity * elapsed
        self.predicted = prediction.copy()
        (
            flow_displacement,
            flow_points,
            flow_quality,
            flow_transform,
        ) = self._flow_displacement(current_gray)
        base_radius = max(45, 2.2 * max(self.width, self.height))
        search_radius = min(
            max(frame_width, frame_height),
            base_radius + self.lost_frames * 8
            + float(np.linalg.norm(self.velocity)) * elapsed,
        )
        candidates = self._candidates(
            frame, prediction, search_radius, clean_frame
        )
        candidates.extend(
            self._outline_matches(frame, prediction, search_radius, clean_frame)
        )
        ranked = []
        motion_prediction = prediction.copy()
        if flow_displacement is not None and flow_quality >= 0.25:
            motion_prediction = 0.5 * prediction + 0.5 * (
                self.center + flow_displacement
            )
        shift = np.round(prediction - self.initial_center).astype(np.int32)
        self.reference_outline = self.template_contour + shift

        for candidate in candidates:
            distance_to_prediction = float(
                np.linalg.norm(candidate["center"] - prediction)
            )
            distance_to_motion = float(
                np.linalg.norm(candidate["center"] - motion_prediction)
            )
            proximity = float(
                np.exp(-distance_to_prediction / max(12, search_radius * 0.35))
            )
            motion = float(
                np.exp(
                    -distance_to_motion
                    / max(8, min(self.width, self.height) * 0.35)
                )
            )
            color_weight = 0.12 if self.lost_frames == 0 else 0.0
            weights = {
                "M": 0.25,
                "P": 0.30,
                "O": 0.45 - color_weight,
                "C": color_weight,
            }
            scores = {
                "M": motion,
                "P": proximity,
                "O": candidate["outline"],
                "C": candidate["color"] if color_weight else 0.0,
            }
            candidate["scores"] = scores
            candidate["distance_to_prediction"] = distance_to_prediction
            candidate["score"] = sum(
                weights[key] * scores[key] for key in weights
            )
            ranked.append(candidate)

        ranked.sort(key=lambda item: item["score"], reverse=True)
        deduplicated = []
        duplicate_radius = max(8, min(self.width, self.height) * 0.25)
        for candidate in ranked:
            duplicate = False
            x, y, width, height = candidate["box"]
            for selected in deduplicated:
                center_distance = np.linalg.norm(
                    candidate["center"] - selected["center"]
                )
                if center_distance > duplicate_radius:
                    continue
                other_x, other_y, other_width, other_height = selected["box"]
                intersection_width = max(
                    0, min(x + width, other_x + other_width) - max(x, other_x)
                )
                intersection_height = max(
                    0, min(y + height, other_y + other_height) - max(y, other_y)
                )
                intersection = intersection_width * intersection_height
                union = width * height + other_width * other_height - intersection
                if union and intersection / union >= 0.35:
                    duplicate = True
                    break
            if not duplicate:
                deduplicated.append(candidate)
        ranked = deduplicated

        self.rejected_candidates = []
        self.candidate_outlines = []
        best = ranked[0] if ranked else None
        margin = (
            best["score"] - ranked[1]["score"]
            if best is not None and len(ranked) > 1
            else 1.0
        )
        accepted = (
            best is not None
            and best["score"] >= self._MATCH_THRESHOLD
            and (margin >= 0.025 or best["score"] >= 0.82)
            and (
                not best["edge_only"]
                or (
                    best["scores"]["O"] >= 0.60
                    and best["scores"]["P"] >= 0.70
                    and best["distance_to_prediction"]
                    <= max(30, min(self.width, self.height) * 0.35)
                    and (
                        (
                            best["scores"]["M"] >= 0.55
                            and margin >= 0.06
                        )
                        or best["score"] >= 0.84
                    )
                )
            )
        )
        if accepted and self.lost_frames >= 12:
            accepted = (
                best["score"] >= 0.88
                and best["scores"]["O"] >= 0.86
                and best["scores"]["P"] >= 0.80
                and margin >= 0.10
                and best["distance_to_prediction"]
                <= max(30, min(self.width, self.height) * 0.75)
            )
        flow_available = (
            flow_transform is not None
            and flow_displacement is not None
            and flow_quality >= 0.35
            and np.linalg.norm(flow_displacement) <= 15
        )
        flow_center = (
            cv2.transform(
                self.center.reshape(1, 1, 2), flow_transform
            )[0, 0]
            if flow_available
            else None
        )
        predicted_step = self.velocity * elapsed
        flow_magnitude = (
            float(np.linalg.norm(flow_displacement))
            if flow_available
            else 0.0
        )
        predicted_magnitude = float(np.linalg.norm(predicted_step))
        flow_matches_prediction = (
            flow_available
            and flow_magnitude < 15
            and (
                flow_magnitude < 0.75
                or predicted_magnitude < 0.75
                or float(np.dot(flow_displacement, predicted_step))
                >= 0.25 * flow_magnitude * predicted_magnitude
            )
            and np.linalg.norm(flow_displacement - predicted_step)
            <= max(7, min(self.width, self.height) * 0.20)
        )
        flow_matches_candidate = (
            accepted
            and flow_available
            and np.linalg.norm(best["aim"] - flow_center)
            <= max(12, min(self.width, self.height) * 0.25)
        )
        flow_is_consistent = flow_matches_candidate or (
            not accepted and flow_matches_prediction
        )

        if accepted or flow_is_consistent:
            previous_outline = self.outline.copy()
            if flow_is_consistent:
                self.outline = cv2.transform(previous_outline, flow_transform)

            if accepted:
                measurement = best["aim"]
                delta = measurement - previous_center
                max_step = max(
                    20,
                    min(self.width, self.height) * 0.35,
                    float(np.linalg.norm(self.velocity)) * elapsed + 10,
                )
                distance = float(np.linalg.norm(delta))
                if distance > max_step:
                    delta *= max_step / distance
                    measurement = previous_center + delta

                new_center = 0.75 * measurement + 0.25 * prediction
                self.confidence = float(best["score"])
                self.match_scores = best["scores"]
                self.visible = True
                measured_velocity = (
                    measurement - previous_center
                ) / elapsed
                self.velocity = (
                    0.65 * self.velocity + 0.35 * measured_velocity
                )
            else:
                new_center = flow_center
                measured_velocity = flow_displacement / elapsed
                self.velocity = (
                    0.65 * self.velocity + 0.35 * measured_velocity
                )
                self.confidence = min(0.75, 0.45 + 0.35 * flow_quality)
                self.match_scores = {
                    "M": flow_quality,
                    "P": 0.0,
                    "O": 0.0,
                    "C": 0.0,
                }
                self.visible = True

            model_correction = new_center - (
                flow_center if flow_is_consistent else previous_center
            )
            self._translate_model(model_correction)
            self.center = new_center
            self.lost_frames = 0
            if flow_is_consistent and flow_points is not None:
                self.previous_points = flow_points.reshape(-1, 1, 2).copy()
                self.previous_points[:, 0, :] += model_correction
            else:
                self.previous_points = self._find_points(
                    frame, self.box, clean_frame, self._edge_feature_mask()
                )
            if self.previous_points is not None and len(self.previous_points) < 6:
                self.previous_points = self._find_points(
                    frame, self.box, clean_frame, self._edge_feature_mask()
                )
            if self.previous_points is not None:
                points = self.previous_points[:, 0, :]
                x, y, width, height = self.box
                x0 = min(x, int(np.floor(np.min(points[:, 0]))))
                y0 = min(y, int(np.floor(np.min(points[:, 1]))))
                x1 = max(
                    x + width,
                    int(np.ceil(np.max(points[:, 0]))) + 1,
                )
                y1 = max(
                    y + height,
                    int(np.ceil(np.max(points[:, 1]))) + 1,
                )
                self.box = (x0, y0, x1 - x0, y1 - y0)
        else:
            self.center = prediction
            self._translate_model(self.center - previous_center)
            self.velocity *= np.exp(-0.35 * elapsed)
            self.confidence *= 0.90
            self.lost_frames += 1
            self.visible = False
            self.match_scores = (
                best["scores"]
                if best is not None
                else {"M": 0.0, "P": 0.0, "O": 0.0, "C": 0.0}
            )
            self.previous_points = None

        self.flow_points = (
            self.previous_points.copy() if self.visible else None
        )
        if not self.visible:
            self.predicted = self.center.copy()
            self.reference_outline = self.outline.copy()
        self.previous_gray = current_gray
        return self.center.copy(), self.confidence, self.box
