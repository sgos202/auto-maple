"""Replay a conservative all-copy Violetta segmentation and identity POC."""

import argparse
import csv
import itertools
from pathlib import Path

import cv2
import numpy as np


_TRACK_COLORS = (
    (255, 100, 30),
    (0, 220, 0),
    (255, 0, 220),
    (0, 180, 255),
)


def detect_instances(frame):
    """Return connected pink-hair masks that are large enough to be copies."""

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (135, 60, 45), (180, 255, 255))
    mask[: round(height * 0.17)] = 0
    mask[round(height * 0.68) :] = 0
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    minimum_area = max(500, round(height * width * 0.0035))
    detections = []

    for label in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[label])
        if area < minimum_area:
            continue
        if not width * 0.07 <= box_width <= width * 0.26:
            continue
        if not height * 0.12 <= box_height <= height * 0.48:
            continue
        aspect = box_width / box_height
        if not 0.5 <= aspect <= 1.5:
            continue

        component = (labels[y : y + box_height, x : x + box_width] == label)
        detections.append(
            {
                "bbox": (x, y, box_width, box_height),
                "center": np.array(
                    (x + box_width / 2, y + box_height / 2),
                    dtype=np.float32,
                ),
                "mask": np.where(component, 255, 0).astype(np.uint8),
                "area": area,
            }
        )

    detections.sort(key=lambda detection: detection["center"][0])
    return detections


def makeup_scores(frame, detections):
    """Measure blue-eye coverage in the face area of each segmented copy."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (85, 100, 60), (135, 255, 255))
    scores = []
    for detection in detections:
        x, y, width, height = detection["bbox"]
        x0 = x + round(width * 0.15)
        x1 = x + round(width * 0.48)
        y0 = y + round(height * 0.68)
        y1 = y + round(height * 0.91)
        eye_region = blue[y0:y1, x0:x1]
        scores.append(
            cv2.countNonZero(eye_region) / max(eye_region.size, 1)
        )
    return scores


def unique_makeup_index(scores, minimum_score=0.06, minimum_margin=0.04):
    """Return a uniquely blue-eyed instance, or None when evidence is weak."""

    if len(scores) != 4:
        return None
    ranking = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    (index, best), (_, second) = ranking[:2]
    if best < minimum_score or best - second < minimum_margin:
        return None
    return index


class MultiCopyIdentityTracker:
    """Associate visible copy masks and retain IDs through short interruptions."""

    MAX_ASSOCIATION_GAP = 45
    MAX_DISPLACEMENT = 0.15
    AMBIGUITY_MARGIN = 0.01

    def __init__(self, frame_width):
        self.frame_width = frame_width
        self.target_id = None
        self.tracks = {}

    def _best_assignment(self, track_ids, detections, frame_index):
        elapsed = {
            track_id: frame_index - self.tracks[track_id]["frame"]
            for track_id in track_ids
        }
        if any(
            frames <= 0 or frames > self.MAX_ASSOCIATION_GAP
            for frames in elapsed.values()
        ):
            return None, False

        predictions = {}
        for track_id in track_ids:
            track = self.tracks[track_id]
            frames = elapsed[track_id]
            predictions[track_id] = (
                track["center"] + track["velocity"] * frames
            )

        scored = []
        for permutation in itertools.permutations(range(len(detections))):
            distances = [
                float(
                    np.linalg.norm(
                        predictions[track_id] - detections[detection_index]["center"]
                    )
                )
                for track_id, detection_index in zip(track_ids, permutation)
            ]
            if max(distances) > self.frame_width * self.MAX_DISPLACEMENT:
                continue
            scored.append((sum(distances), permutation))

        if not scored:
            return None, False
        scored.sort(key=lambda item: item[0])
        if (
            len(scored) > 1
            and scored[1][0] - scored[0][0]
            < self.frame_width * self.AMBIGUITY_MARGIN
        ):
            return None, True
        return dict(zip(track_ids, scored[0][1])), False

    def _remember(self, assignments, detections, frame_index):
        for track_id, detection_index in assignments.items():
            center = detections[detection_index]["center"]
            previous = self.tracks.get(track_id)
            velocity = np.zeros(2, dtype=np.float32)
            if previous is not None:
                elapsed = frame_index - previous["frame"]
                if elapsed > 0:
                    velocity = (center - previous["center"]) / elapsed
            self.tracks[track_id] = {
                "center": center.copy(),
                "velocity": velocity,
                "frame": frame_index,
            }

    def update(self, detections, frame_index, makeup_index=None):
        """Return {track_id: detection_index} and whether assignment is ambiguous."""

        if len(detections) != 4:
            return {}, False

        if not self.tracks:
            if makeup_index is None:
                return {}, False
            self.target_id = makeup_index + 1
            assignments = {index + 1: index for index in range(4)}
            self._remember(assignments, detections, frame_index)
            return assignments, False

        assignments = {}
        if makeup_index is not None and self.target_id is not None:
            assignments[self.target_id] = makeup_index

        remaining_ids = [
            track_id
            for track_id in sorted(self.tracks)
            if track_id not in assignments
        ]
        remaining_detections = [
            index for index in range(4) if index not in assignments.values()
        ]
        if len(remaining_ids) == len(remaining_detections):
            association, ambiguous = self._best_assignment(
                remaining_ids,
                [detections[index] for index in remaining_detections],
                frame_index,
            )
            if association is None:
                if assignments:
                    self._remember(assignments, detections, frame_index)
                return assignments, ambiguous
            assignments.update(
                {
                    track_id: remaining_detections[local_index]
                    for track_id, local_index in association.items()
                }
            )

        self._remember(assignments, detections, frame_index)
        return assignments, False


def _draw(
    frame,
    detections,
    assignments,
    target_id,
    phase,
    target_index,
):
    output = frame.copy()
    id_by_index = {
        detection_index: track_id
        for track_id, detection_index in assignments.items()
    }

    for index, detection in enumerate(detections):
        track_id = id_by_index.get(index)
        is_target = track_id == target_id or (
            target_index is not None and index == target_index
        )
        color = (0, 255, 255) if is_target else (
            _TRACK_COLORS[(track_id - 1) % len(_TRACK_COLORS)]
            if track_id is not None
            else (170, 170, 170)
        )
        x, y, width, height = detection["bbox"]
        contours, _ = cv2.findContours(
            detection["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [
            contour + np.array((x, y), dtype=np.int32)
            for contour in contours
        ]
        cv2.drawContours(output, contours, -1, color, 3)
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            color,
            3 if is_target else 1,
        )
        if is_target and target_index == index:
            label = f"TARGET ID {track_id or target_id} (INITIAL CUE)"
        elif is_target:
            label = f"TARGET ID {track_id} (MOTION ASSOCIATION)"
        else:
            label = f"ID {track_id}" if track_id is not None else "ID UNKNOWN"
        cv2.putText(
            output,
            label,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(output, (0, 0), (output.shape[1], 48), (20, 20, 20), -1)
    cv2.putText(
        output,
        phase,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _target_snapshot(frame, detection, label):
    output = frame.copy()
    x, y, width, height = detection["bbox"]
    cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 255), 6)
    cv2.rectangle(output, (0, 0), (output.shape[1], 52), (20, 20, 20), -1)
    cv2.putText(
        output,
        label,
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _write_target_comparison(path, start_frame, last_tracked_frame):
    if start_frame is None or last_tracked_frame is None:
        raise RuntimeError(
            "Could not create comparison without a start and motion-tracked frame."
        )
    comparison = np.hstack((start_frame, last_tracked_frame))
    if not cv2.imwrite(str(path), comparison):
        raise RuntimeError(f"Could not write target comparison image: {path}")


def run_video(video_path, output_path, metrics_path, comparison_path):
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

    metrics_stream = None
    metrics_writer = None
    if metrics_path:
        try:
            metrics_stream = open(
                metrics_path, "w", newline="", encoding="utf-8"
            )
        except OSError:
            capture.release()
            if writer:
                writer.release()
            raise
        metrics_writer = csv.DictWriter(
            metrics_stream,
            fieldnames=(
                "frame",
                "time_seconds",
                "visible_instances",
                "assigned_ids",
                "target_slot",
                "target_makeup_score",
                "target_state",
                "phase",
            ),
        )
        metrics_writer.writeheader()

    identities = MultiCopyIdentityTracker(width)
    frame_index = 0
    initialized = False
    target_slot = None
    target_score = 0.0
    seen_unique_targets = []
    four_visible_frames = 0
    ambiguous_frames = 0
    start_snapshot = None
    last_tracked_snapshot = None
    last_tracked_frame = None
    last_tracked_slot = None
    slot_centers = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detect_instances(frame)
            scores = (
                makeup_scores(frame, detections)
                if not initialized and len(detections) == 4
                else []
            )
            makeup_index = (
                unique_makeup_index(scores)
                if not initialized
                else None
            )
            if makeup_index is not None:
                target_slot = makeup_index + 1
                target_score = scores[makeup_index]
                seen_unique_targets.append((frame_index, target_slot, target_score))
                snapshot_label = (
                    f"TARGET ID {identities.target_id or makeup_index + 1} - "
                    f"START SLOT {target_slot} (INITIAL REVEAL ONLY)"
                )
                snapshot = _target_snapshot(
                    frame, detections[makeup_index], snapshot_label
                )
                if start_snapshot is None:
                    start_snapshot = snapshot
                slot_centers = [
                    detection["center"][0] / width
                    for detection in detections
                ]

            assignments, ambiguous_assignment = identities.update(
                detections, frame_index, makeup_index
            )
            target_detection_index = (
                assignments.get(identities.target_id)
                if identities.target_id is not None
                else None
            )
            if makeup_index is not None:
                target_state = "confirmed_by_makeup"
            elif identities.target_id in assignments:
                target_state = "associated_by_motion"
            else:
                target_state = "unknown"
            if initialized and target_detection_index is not None:
                target_detection = detections[target_detection_index]
                center_x = target_detection["center"][0] / width
                last_tracked_slot = min(
                    range(len(slot_centers)),
                    key=lambda index: abs(slot_centers[index] - center_x),
                ) + 1
                last_tracked_frame = frame_index
                last_tracked_snapshot = _target_snapshot(
                    frame,
                    target_detection,
                    f"LAST MOTION-SUPPORTED BOX - {frame_index / fps:.2f}s "
                    f"(SLOT {last_tracked_slot}; NOT FINAL)",
                )
            if len(detections) == 4:
                four_visible_frames += 1
            if ambiguous_assignment:
                ambiguous_frames += 1

            if not initialized:
                phase = (
                    "WAITING FOR FOUR COPIES AND DISTINCTIVE MAKEUP"
                    if makeup_index is None
                    else f"INITIALIZED: TARGET ID {identities.target_id} IN SLOT {target_slot}"
                )
                initialized = makeup_index is not None
            elif identities.target_id in assignments:
                phase = (
                    f"TARGET ID {identities.target_id} MOTION-ASSOCIATED; "
                    "NOT MAKEUP-REIDENTIFIED"
                )
            elif ambiguous_assignment:
                phase = "TARGET IDENTITY UNKNOWN - MOTION ASSOCIATION AMBIGUOUS"
            elif len(detections) != 4:
                phase = (
                    f"TARGET IDENTITY UNKNOWN - {len(detections)} COPY MASKS; "
                    "NO TARGET BOX"
                )
                if initialized:
                    ambiguous_frames += 1
            else:
                phase = "TARGET IDENTITY UNKNOWN - NO TARGET BOX"
                if initialized:
                    ambiguous_frames += 1

            annotated = _draw(
                frame,
                detections,
                assignments,
                identities.target_id,
                phase,
                makeup_index if target_state == "confirmed_by_makeup" else None,
            )
            if writer:
                writer.write(annotated)
            if metrics_writer:
                metrics_writer.writerow(
                    {
                        "frame": frame_index,
                        "time_seconds": f"{frame_index / fps:.4f}",
                        "visible_instances": len(detections),
                        "assigned_ids": ";".join(
                            f"{track_id}:{detection_index + 1}"
                            for track_id, detection_index in sorted(
                                assignments.items()
                            )
                        ),
                        "target_slot": target_slot if makeup_index is not None else "",
                        "target_makeup_score": (
                            f"{target_score:.4f}" if makeup_index is not None else ""
                        ),
                        "target_state": target_state,
                        "phase": phase,
                    }
                )
            frame_index += 1
    finally:
        capture.release()
        if writer:
            writer.release()
        if metrics_stream:
            metrics_stream.close()

    if frame_index == 0:
        raise RuntimeError(f"The supplied video contained no frames: {video_path}")
    if not seen_unique_targets:
        raise RuntimeError(
            "Could not confidently identify the blue-eyed Violetta in the video."
        )
    if comparison_path:
        _write_target_comparison(
            comparison_path,
            start_snapshot,
            last_tracked_snapshot,
        )

    first = seen_unique_targets[0]
    print(f"[POC] Frames processed: {frame_index} ({frame_index / fps:.2f}s)")
    print(f"[POC] Frames with four separated copy masks: {four_visible_frames}")
    print(
        f"[POC] Target ID {identities.target_id}: initial slot {first[1]} "
        "(initial reveal cue only)."
    )
    print(
        f"[POC] Makeup cue used only at {first[0] / fps:.2f}s; "
        "no post-shuffle appearance cue was used."
    )
    if last_tracked_frame is not None:
        print(
            f"[POC] Last motion-supported target box: "
            f"{last_tracked_frame / fps:.2f}s, slot {last_tracked_slot}; "
            "this is not a final-position estimate."
        )
    else:
        print("[POC] No target motion association was confirmed after initialization.")
    if ambiguous_frames:
        print(f"[POC] Frames explicitly left ambiguous: {ambiguous_frames}")
    if output_path:
        print(f"[POC] Annotated replay: {output_path}")
    if metrics_path:
        print(f"[POC] Frame metrics: {metrics_path}")
    if comparison_path:
        print(f"[POC] Start/last-tracked target boxes: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay an all-copy segmentation and target-identity POC."
    )
    parser.add_argument("--video", required=True, help="Input gameplay recording.")
    parser.add_argument("--output", help="Optional annotated MP4 output.")
    parser.add_argument("--metrics", help="Optional per-frame CSV output.")
    parser.add_argument(
        "--comparison",
        help="Optional side-by-side image boxing the target at start and end.",
    )
    args = parser.parse_args()
    run_video(args.video, args.output, args.metrics, args.comparison)


if __name__ == "__main__":
    main()
