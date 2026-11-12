"""Small YOLO-seg dataset and motion-only identity-tracking POC."""

import argparse
import csv
import itertools
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def prepare_frames(video_path, dataset_dir, interval, end_seconds=10.0):
    video_path = Path(video_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    frames_dir = dataset_dir / "frames"
    if frames_dir.exists() and any(frames_dir.iterdir()):
        raise FileExistsError(
            f"Frame folder is not empty; refusing to overwrite: {frames_dir}"
        )
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid frame rate: {video_path}")
    stride = max(1, round(fps * interval))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    manifest = []
    frame_index = 0
    sample_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            if timestamp > end_seconds:
                break
            if frame_index % stride == 0:
                name = f"frame_{sample_index:05d}.jpg"
                image_path = frames_dir / name
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError(f"Could not write sampled frame: {image_path}")
                manifest.append(
                    {
                        "image": name,
                        "frame": frame_index,
                        "time_seconds": round(timestamp, 4),
                    }
                )
                sample_index += 1
            frame_index += 1
    finally:
        capture.release()
    if not manifest:
        raise RuntimeError(f"No frames were extracted from {video_path}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "source_video": str(video_path),
                "fps": fps,
                "frame_size": [frame_width, frame_height],
                "interval_seconds": interval,
                "end_seconds": end_seconds,
                "frames": manifest,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return len(manifest)


class ViolettaLabeler:
    """Draw all visible Violetta masks as polygons on sampled frames."""

    COLORS = ("#ff5555", "#55dd55", "#55aaff", "#ffaa33")

    def __init__(self, dataset_dir):
        import tkinter as tk

        self.tk = tk
        self.dataset_dir = Path(dataset_dir).resolve()
        self.frames_dir = self.dataset_dir / "frames"
        self.manifest = _read_json(self.dataset_dir / "manifest.json")
        self.annotations_path = self.dataset_dir / "annotations.json"
        self.annotations = {}
        if self.annotations_path.exists():
            saved = _read_json(self.annotations_path)
            if saved.get("source_video") != self.manifest["source_video"]:
                raise ValueError("Existing annotations belong to another video.")
            self.annotations = {
                item["image"]: item for item in saved.get("annotations", [])
            }
        self.position = 0
        self.instances = []
        self.points = []
        self.scale = 1.0
        self.root = tk.Tk()
        self.root.title("Violetta all-copy polygon labeler")
        self.root.bind("<Return>", lambda _event: self.finish_instance())
        self.root.bind("<KeyPress-d>", lambda _event: self.finish_frame())
        self.root.bind("<KeyPress-n>", lambda _event: self.next_frame())
        self.root.bind("<KeyPress-p>", lambda _event: self.previous_frame())
        self.root.bind("<KeyPress-u>", lambda _event: self.undo_instance())
        self.root.bind("<Control-s>", lambda _event: self.save())
        tk.Label(
            self.root,
            text=(
                "Click polygon points; Enter closes one copy; repeat for all visible "
                "copies; D saves frame; N skip; P previous; right-click undo point."
            ),
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=4)
        self.status = tk.StringVar()
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(
            fill=tk.X, padx=8, pady=4
        )
        self.canvas = tk.Canvas(self.root, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=8, pady=4)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-3>", self.undo_point)
        self.show_frame()

    def show_frame(self):
        entry = self.manifest["frames"][self.position]
        self.image = cv2.imread(str(self.frames_dir / entry["image"]))
        if self.image is None:
            raise RuntimeError(f"Could not read sampled frame {entry['image']}")
        height, width = self.image.shape[:2]
        self.scale = min(1.0, 1150 / width, 700 / height)
        display = cv2.resize(
            self.image,
            (round(width * self.scale), round(height * self.scale)),
            interpolation=cv2.INTER_AREA,
        )
        ok, encoded = cv2.imencode(".png", display)
        if not ok:
            raise RuntimeError("Could not encode frame for labeler.")
        import base64

        self.photo = self.tk.PhotoImage(
            data=base64.b64encode(encoded).decode("ascii"), format="PNG"
        )
        self.canvas.configure(width=display.shape[1], height=display.shape[0])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=self.tk.NW)
        saved = self.annotations.get(entry["image"])
        self.instances = [item[:] for item in saved["polygons"]] if saved else []
        self.points = []
        self._draw_instances()
        self._draw_points()
        self._status()

    def _status(self):
        entry = self.manifest["frames"][self.position]
        self.status.set(
            f"{self.position + 1}/{len(self.manifest['frames'])} "
            f"{entry['time_seconds']:.2f}s | "
            f"{len(self.instances)} polygons | {len(self.points)} current points"
        )

    def _draw_instances(self):
        for index, polygon in enumerate(self.instances):
            coords = [
                (round(x * self.scale), round(y * self.scale)) for x, y in polygon
            ]
            coords.append(coords[0])
            self.canvas.create_line(
                *[v for point in coords for v in point],
                fill=self.COLORS[index % len(self.COLORS)],
                width=2,
            )

    def _draw_points(self):
        self.canvas.delete("draft")
        coords = [
            (round(x * self.scale), round(y * self.scale)) for x, y in self.points
        ]
        if len(coords) > 1:
            self.canvas.create_line(
                *[v for point in coords for v in point],
                fill="#ffffff",
                width=2,
                tags="draft",
            )
        for x, y in coords:
            self.canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3,
                fill="#ffffff", outline="#111111", tags="draft"
            )
        self._status()

    def click(self, event):
        if event.x >= self.canvas.winfo_width() or event.y >= self.canvas.winfo_height():
            return
        self.points.append((round(event.x / self.scale, 2), round(event.y / self.scale, 2)))
        self._draw_points()

    def undo_point(self, _event):
        if self.points:
            self.points.pop()
            self._draw_points()

    def undo_instance(self):
        if self.instances:
            self.instances.pop()
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.photo, anchor=self.tk.NW)
            self._draw_instances()
            self._draw_points()

    def finish_instance(self):
        if len(self.points) < 3:
            self.status.set("A mask polygon needs at least 3 points.")
            return
        self.instances.append(self.points)
        self.points = []
        self._draw_instances()
        self._draw_points()

    def finish_frame(self):
        if self.points:
            self.status.set("Press Enter to close the current polygon first.")
            return
        entry = self.manifest["frames"][self.position]
        self.annotations[entry["image"]] = {
            "image": entry["image"],
            "frame": entry["frame"],
            "time_seconds": entry["time_seconds"],
            "polygons": self.instances,
        }
        self.save()
        self.next_frame()

    def next_frame(self):
        if self.position + 1 < len(self.manifest["frames"]):
            self.position += 1
            self.show_frame()

    def previous_frame(self):
        if self.position:
            self.position -= 1
            self.show_frame()

    def save(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_video": self.manifest["source_video"],
            "frame_size": self.manifest["frame_size"],
            "annotations": [
                self.annotations[key] for key in sorted(self.annotations)
            ],
        }
        temporary = self.annotations_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        temporary.replace(self.annotations_path)

    def run(self):
        self.root.mainloop()


def _split_for_time(timestamp, block_seconds):
    return "val" if int(timestamp / block_seconds) % 2 == 1 else "train"


def export_dataset(dataset_dir, block_seconds=3.0):
    dataset_dir = Path(dataset_dir).resolve()
    manifest = _read_json(dataset_dir / "manifest.json")
    annotations_data = _read_json(dataset_dir / "annotations.json")
    if annotations_data.get("source_video") != manifest["source_video"]:
        raise ValueError("Annotations do not match the sampled source video.")
    annotations = {
        item["image"]: item for item in annotations_data.get("annotations", [])
    }
    if not annotations:
        raise ValueError("Label at least one frame before exporting.")
    splits = {"train": 0, "val": 0}
    for image_name, annotation in annotations.items():
        split = _split_for_time(annotation["time_seconds"], block_seconds)
        image = cv2.imread(str(dataset_dir / "frames" / image_name))
        if image is None:
            raise RuntimeError(f"Missing sampled frame: {image_name}")
        height, width = image.shape[:2]
        label_lines = []
        for polygon in annotation["polygons"]:
            if len(polygon) < 3:
                raise ValueError(f"Invalid polygon in {image_name}")
            coords = []
            for x, y in polygon:
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError(f"Out-of-frame polygon point in {image_name}")
                coords.extend((x / width, y / height))
            label_lines.append(
                "0 " + " ".join(f"{coordinate:.6f}" for coordinate in coords)
            )
        image_output = dataset_dir / "images" / split / image_name
        label_output = dataset_dir / "labels" / split / f"{Path(image_name).stem}.txt"
        image_output.parent.mkdir(parents=True, exist_ok=True)
        label_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset_dir / "frames" / image_name, image_output)
        label_output.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        splits[split] += 1
    if not splits["train"] or not splits["val"]:
        raise ValueError(
            "Temporal split needs labeled frames in both train and val. "
            "Label samples from at least two 3-second blocks."
        )
    config = dataset_dir / "dataset.yaml"
    config.write_text(
        f"path: {json.dumps(dataset_dir.as_posix())}\n"
        "train: images/train\nval: images/val\nnames:\n  0: violetta\n",
        encoding="utf-8",
    )
    return config, splits


def train_model(dataset_yaml, output_dir, epochs=30, image_size=640):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "YOLO training requires Ultralytics. Install it in a separate "
            "environment with `python -m pip install ultralytics`."
        ) from error
    model = YOLO("yolo26n-seg.pt")
    model.train(
        data=str(Path(dataset_yaml).resolve()),
        epochs=epochs,
        imgsz=image_size,
        device="cpu",
        workers=0,
        project=str(Path(output_dir).resolve()),
        name="violetta",
        exist_ok=True,
    )


def _mask_features(mask):
    binary = (mask > 0).astype(np.uint8) * 255
    moments = cv2.moments(binary)
    if moments["m00"] == 0:
        raise ValueError("Cannot track an empty segmentation mask.")
    center = np.array(
        (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
        dtype=np.float32,
    )
    ys, xs = np.nonzero(binary)
    width = max(int(xs.max() - xs.min() + 1), 1)
    height = max(int(ys.max() - ys.min() + 1), 1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    oriented_width, oriented_height = cv2.minAreaRect(contour)[1]
    aspect = max(oriented_width, oriented_height) / max(
        min(oriented_width, oriented_height), 1.0
    )
    return {
        "center": center,
        "area": float(cv2.countNonZero(binary)),
        "aspect": aspect,
        "contour": contour,
        "bbox": (int(xs.min()), int(ys.min()), width, height),
        "mask": binary,
    }


class MotionMaskTracker:
    """Associate all-copy masks using predicted motion and normalized shape."""

    MAX_MOTION_PER_FRAME = 0.04
    MAX_GAP = 12
    AMBIGUITY_MARGIN = 0.12

    def __init__(self, frame_width, target_slot=2):
        self.frame_width = frame_width
        self.target_slot = target_slot
        self.tracks = {}

    def initialize(self, detections, frame_index=0):
        if len(detections) != 4:
            return False
        ordered = sorted(detections, key=lambda item: item["center"][0])
        self.tracks = {
            index + 1: {
                **detection,
                "velocity": np.zeros(2, dtype=np.float32),
                "frame": frame_index,
            }
            for index, detection in enumerate(ordered)
        }
        return True

    def _cost(self, track, detection, frame_index):
        elapsed = frame_index - track["frame"]
        if elapsed <= 0 or elapsed > self.MAX_GAP:
            return math.inf
        prediction = track["center"] + track["velocity"] * elapsed
        distance = float(np.linalg.norm(prediction - detection["center"]))
        motion_limit = self.frame_width * self.MAX_MOTION_PER_FRAME * elapsed
        if distance > motion_limit:
            return math.inf
        motion_cost = distance / max(motion_limit, 1)
        scale_cost = min(abs(math.log(detection["area"] / track["area"])), 2.0) / 2
        aspect_cost = min(
            abs(math.log(detection["aspect"] / track["aspect"])), 1.0
        )
        shape_cost = min(
            cv2.matchShapes(
                track["contour"],
                detection["contour"],
                cv2.CONTOURS_MATCH_I1,
                0.0,
            ),
            1.0,
        )
        return motion_cost + 0.2 * scale_cost + 0.1 * aspect_cost + 0.15 * shape_cost

    def update(self, detections, frame_index):
        detections = sorted(detections, key=lambda item: item["center"][0])
        track_ids = sorted(self.tracks)
        if len(track_ids) != 4 or len(detections) != 4:
            return {}, False
        scored = []
        for permutation in itertools.permutations(range(4)):
            costs = [
                self._cost(
                    self.tracks[track_id], detections[detection_index], frame_index
                )
                for track_id, detection_index in zip(track_ids, permutation)
            ]
            if all(math.isfinite(cost) for cost in costs):
                scored.append((sum(costs) / 4, permutation))
        if not scored:
            return {}, False
        scored.sort(key=lambda result: result[0])
        if (
            len(scored) > 1
            and scored[1][0] - scored[0][0] < self.AMBIGUITY_MARGIN
        ):
            return {}, True
        assignments = dict(zip(track_ids, scored[0][1]))
        for track_id, detection_index in assignments.items():
            detection = detections[detection_index]
            previous = self.tracks[track_id]
            elapsed = frame_index - previous["frame"]
            velocity = (detection["center"] - previous["center"]) / elapsed
            self.tracks[track_id] = {
                **detection,
                "velocity": velocity,
                "frame": frame_index,
            }
        return assignments, False


def _yolo_detections(model, frame):
    result = model.predict(frame, verbose=False, device="cpu")[0]
    detections = []
    if result.masks is None:
        return detections
    for polygon in result.masks.xy:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if len(polygon) < 3:
            continue
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
        try:
            detections.append(_mask_features(mask))
        except ValueError:
            continue
    return sorted(detections, key=lambda item: item["center"][0])


def track_video(
    model_path,
    video_path,
    output_path,
    metrics_path,
    target_slot=2,
    initialization_seconds=3.0,
):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "YOLO tracking requires Ultralytics in this Python environment."
        ) from error
    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    tracker = MotionMaskTracker(width, target_slot)
    with Path(metrics_path).open("w", newline="", encoding="utf-8") as stream:
        metrics = csv.DictWriter(
            stream,
            fieldnames=("frame", "time_seconds", "visible", "target_state", "cx", "cy"),
        )
        metrics.writeheader()
        frame_index = 0
        initialized = False
        initialization_frame = None
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = _yolo_detections(model, frame)
                ambiguous = False
                if not initialized:
                    if frame_index / fps <= initialization_seconds:
                        initialized = tracker.initialize(detections, frame_index)
                        if initialized:
                            initialization_frame = frame_index
                    assignments = (
                        {index + 1: index for index in range(4)}
                        if initialized
                        else {}
                    )
                else:
                    assignments, ambiguous = tracker.update(detections, frame_index)
                target_detection = (
                    detections[assignments[tracker.target_slot]]
                    if tracker.target_slot in assignments
                    else None
                )
                for detection_index, detection in enumerate(detections):
                    track_id = next(
                        (
                            identity
                            for identity, assigned_index in assignments.items()
                            if assigned_index == detection_index
                        ),
                        None,
                    )
                    color = (0, 255, 255) if track_id == tracker.target_slot else (255, 120, 0)
                    cv2.drawContours(
                        frame,
                        [detection["contour"]],
                        -1,
                        color,
                        2,
                        offset=(0, 0),
                    )
                    x, y, box_width, box_height = detection["bbox"]
                    cv2.rectangle(
                        frame, (x, y), (x + box_width, y + box_height), color, 2
                    )
                    if track_id is not None:
                        cv2.putText(
                            frame,
                            f"ID {track_id}",
                            (x, max(y - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )
                state = (
                    "INITIAL TARGET SLOT"
                            if frame_index == initialization_frame
                    else "MOTION-ASSOCIATED"
                    if target_detection is not None
                    else "UNKNOWN"
                )
                if target_detection is not None:
                    cx, cy = target_detection["center"]
                    cv2.putText(
                        frame,
                        f"TARGET {state}",
                        (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cx = cy = ""
                    cv2.putText(
                        frame,
                        f"TARGET {state} - NO BOX",
                        (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                metrics.writerow(
                    {
                        "frame": frame_index,
                        "time_seconds": f"{frame_index / fps:.4f}",
                        "visible": len(detections),
                        "target_state": state,
                        "cx": cx,
                        "cy": cy,
                    }
                )
                writer.write(frame)
                frame_index += 1
        finally:
            capture.release()
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Sample training frames.")
    prepare.add_argument("--video", required=True)
    prepare.add_argument("--dataset", default="assets/violetta_training")
    prepare.add_argument("--interval", type=float, default=0.3)
    prepare.add_argument(
        "--end-seconds",
        type=float,
        default=10.0,
        help="Stop before the post-selection answer reveal (default: 10 seconds).",
    )

    label = commands.add_parser("label", help="Polygon-label all visible copies.")
    label.add_argument("--dataset", default="assets/violetta_training")

    export = commands.add_parser("export", help="Export YOLO-seg train/val data.")
    export.add_argument("--dataset", default="assets/violetta_training")
    export.add_argument("--block-seconds", type=float, default=3.0)

    train = commands.add_parser("train", help="Fine-tune a small YOLO-seg model on CPU.")
    train.add_argument("--dataset", default="assets/violetta_training/dataset.yaml")
    train.add_argument("--output", default="runs/violetta")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--image-size", type=int, default=640)

    track = commands.add_parser("track", help="Track identities without later face cues.")
    track.add_argument("--model", required=True)
    track.add_argument("--video", required=True)
    track.add_argument("--output", default="violetta-yolo-poc.mp4")
    track.add_argument("--metrics", default="violetta-yolo-poc.csv")
    track.add_argument("--target-slot", type=int, choices=range(1, 5), default=2)
    track.add_argument(
        "--initialization-seconds",
        type=float,
        default=3.0,
        help="Only initialize during the opening reveal (default: first 3 seconds).",
    )
    args = parser.parse_args()

    if args.command == "prepare":
        if args.interval <= 0:
            parser.error("--interval must be positive")
        if args.end_seconds <= 0:
            parser.error("--end-seconds must be positive")
        print(
            "Extracted "
            f"{prepare_frames(args.video, args.dataset, args.interval, args.end_seconds)} "
            "frames."
        )
    elif args.command == "label":
        ViolettaLabeler(args.dataset).run()
    elif args.command == "export":
        config, splits = export_dataset(args.dataset, args.block_seconds)
        print(f"Exported {splits}; dataset config: {config}")
    elif args.command == "train":
        train_model(args.dataset, args.output, args.epochs, args.image_size)
    elif args.command == "track":
        track_video(
            args.model,
            args.video,
            args.output,
            args.metrics,
            args.target_slot,
            args.initialization_seconds,
        )


if __name__ == "__main__":
    main()
