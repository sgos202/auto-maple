"""Label arbitrary target outlines on sampled challenge-video frames."""

import argparse
import base64
import json
import tkinter as tk
from pathlib import Path

import cv2


class PolygonLabeler:
    POINT_COLORS = ("#ff5555", "#55dd55", "#55aaff", "#ffaa33", "#dd66ff", "#44dddd")

    def __init__(self, video_path, output_path, interval, label_mode="polygon"):
        self.video_path = Path(video_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.label_mode = label_mode
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.fps <= 0 or self.frame_count <= 0:
            self.capture.release()
            raise RuntimeError(f"Video has invalid timing or dimensions: {self.video_path}")

        stride = max(1, round(self.fps * interval))
        self.sample_frames = list(range(0, self.frame_count, stride))
        if self.sample_frames[-1] != self.frame_count - 1:
            self.sample_frames.append(self.frame_count - 1)

        self.labels = self._load_labels()
        self.sample_position = 0
        self.vertices = []
        self.display_scale = 1.0
        self.root = tk.Tk()
        self.root.title(
            "Shape point labeler" if self.label_mode == "points" else "Shape outline labeler"
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress-n>", lambda _event: self.next_frame())
        self.root.bind("<KeyPress-p>", lambda _event: self.previous_frame())
        self.root.bind("<KeyPress-Return>", lambda _event: self.finish_polygon())
        self.root.bind("<Control-s>", lambda _event: self.save())

        if self.label_mode == "points":
            instructions = (
                "Click the center first, then the same distinctive features in order "
                "on each frame. Enter: save  Right-click: undo  N: skip  "
                "P: previous  Ctrl+S: save"
            )
        else:
            instructions = (
                "Click around the target outline, then Enter to save. "
                "Right-click removes the last point. N: skip  P: previous  Ctrl+S: save"
            )
        tk.Label(
            self.root,
            text=instructions,
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(8, 4))

        self.status = tk.StringVar()
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(
            fill=tk.X, padx=8, pady=4
        )
        self.canvas = tk.Canvas(self.root, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=8, pady=4)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=(4, 8))
        tk.Button(controls, text="Previous", command=self.previous_frame).pack(
            side=tk.LEFT
        )
        tk.Button(controls, text="Skip", command=self.next_frame).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        finish_text = "Save points" if self.label_mode == "points" else "Finish outline"
        tk.Button(controls, text=finish_text, command=self.finish_polygon).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        tk.Button(controls, text="Save", command=self.save).pack(side=tk.RIGHT)

        self.show_frame()

    def _load_labels(self):
        if not self.output_path.exists():
            return {}
        with self.output_path.open("r", encoding="utf-8") as label_file:
            data = json.load(label_file)
        expected_type = "tracked_points" if self.label_mode == "points" else "polygon_vertices"
        stored_type = data.get("label_type")
        if stored_type != expected_type and not (
            self.label_mode == "polygon" and stored_type is None
        ):
            raise ValueError(
                f"Label file contains {stored_type} labels, not {expected_type}: "
                f"{self.output_path}"
            )
        if data.get("source_video") != str(self.video_path):
            raise ValueError(
                f"Label file belongs to a different video: {self.output_path}"
            )
        if data.get("frame_size") != [self.frame_width, self.frame_height]:
            raise ValueError(f"Label file has different video dimensions: {self.output_path}")
        annotations = {int(item["frame"]): item for item in data.get("annotations", [])}
        if self.label_mode == "points":
            if any("points" not in item for item in annotations.values()):
                raise ValueError(f"Tracked-point labels are malformed: {self.output_path}")
            point_counts = {
                len(item.get("points", [])) for item in annotations.values()
            }
            if len(point_counts) > 1:
                raise ValueError(
                    f"Tracked-point labels must have the same point count per frame: "
                    f"{self.output_path}"
                )
        return annotations

    def _frame_number(self):
        return self.sample_frames[self.sample_position]

    def show_frame(self):
        frame_number = self._frame_number()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"Could not read video frame {frame_number}.")

        self.display_scale = min(
            1.0,
            1100 / self.frame_width,
            720 / self.frame_height,
        )
        display_size = (
            round(self.frame_width * self.display_scale),
            round(self.frame_height * self.display_scale),
        )
        display = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
        if not self._show_image(display):
            raise RuntimeError("Could not encode the video frame for display.")

        self.vertices = []
        self.canvas.delete("annotation")
        annotation = self.labels.get(frame_number)
        if annotation:
            if self.label_mode == "points":
                self._draw_points(annotation["points"], "annotation")
            else:
                self._draw_polygon(annotation["vertices"], "#00ff00", "annotation")
        timestamp = frame_number / self.fps
        label_status = "labeled" if annotation else "not labeled"
        self.status.set(
            f"Frame {self.sample_position + 1}/{len(self.sample_frames)} "
            f"at {timestamp:.2f}s — {label_status} — {len(self.labels)} labels"
        )

    def _show_image(self, image):
        encoded, png = cv2.imencode(".png", image)
        if not encoded:
            return False
        image_data = base64.b64encode(png.tobytes()).decode("ascii")
        self.photo = tk.PhotoImage(data=image_data, format="PNG")
        self.canvas.configure(width=image.shape[1], height=image.shape[0])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        return True

    def _draw_polygon(self, vertices, color, tag):
        points = [
            (round(x * self.display_scale), round(y * self.display_scale))
            for x, y in vertices
        ]
        if tag == "annotation" and len(points) >= 3:
            points.append(points[0])
        if len(points) >= 2:
            self.canvas.create_line(
                *[coordinate for point in points for coordinate in point],
                fill=color,
                width=2,
                tags=tag,
            )
        for x, y in points:
            self.canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, fill=color, outline=color, tags=tag
            )

    def _refresh_draft(self):
        self.canvas.delete("draft")
        if self.label_mode == "points":
            self._draw_points(self.vertices, "draft")
        else:
            self._draw_polygon(self.vertices, "#00aaff", "draft")

    def _draw_points(self, points, tag):
        for index, (x, y) in enumerate(points):
            x = round(x * self.display_scale)
            y = round(y * self.display_scale)
            color = self.POINT_COLORS[index % len(self.POINT_COLORS)]
            self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5, fill=color, outline="white", tags=tag
            )
            point_id = "C" if index == 0 else str(index)
            self.canvas.create_text(
                x + 10,
                y - 10,
                text=point_id,
                fill=color,
                font=("TkDefaultFont", 10, "bold"),
                tags=tag,
            )

    def on_left_click(self, event):
        if event.x >= self.canvas.winfo_width() or event.y >= self.canvas.winfo_height():
            return
        self.vertices.append(
            (event.x / self.display_scale, event.y / self.display_scale)
        )
        self._refresh_draft()
        if self.label_mode == "points":
            self.status.set(
                f"{len(self.vertices)} tracked points (C is the center); press Enter to save."
            )
        else:
            self.status.set(f"{len(self.vertices)} outline points; press Enter when done.")

    def on_right_click(self, _event):
        if self.vertices:
            self.vertices.pop()
            self._refresh_draft()
            label = "tracked points" if self.label_mode == "points" else "outline points"
            self.status.set(f"{len(self.vertices)} {label}.")
            return
        frame_number = self._frame_number()
        if frame_number in self.labels:
            del self.labels[frame_number]
            self.save()
            self.show_frame()

    def finish_polygon(self):
        if len(self.vertices) < 3:
            if self.label_mode == "points":
                self.status.set("Add the center and at least 2 distinctive points before saving.")
                return
            self.status.set("Add at least 3 points around the outline before saving.")
            return
        frame_number = self._frame_number()
        coordinates = [[round(x, 2), round(y, 2)] for x, y in self.vertices]
        if self.label_mode == "points":
            expected_counts = {
                len(annotation["points"])
                for existing_frame, annotation in self.labels.items()
                if existing_frame != frame_number
            }
            if expected_counts and len(coordinates) not in expected_counts:
                expected_count = next(iter(expected_counts))
                self.status.set(
                    f"Use {expected_count} points to preserve point IDs "
                    f"(center first, then features in the same order)."
                )
                return
            annotation = {"points": coordinates}
        else:
            annotation = {"vertices": coordinates}
        annotation.update(
            {
                "frame": frame_number,
                "time_seconds": frame_number / self.fps,
            }
        )
        self.labels[frame_number] = annotation
        self.save()
        if self.sample_position + 1 < len(self.sample_frames):
            self.sample_position += 1
            self.show_frame()
        else:
            self.show_frame()
            self.status.set("End of video. Labels are saved; use Previous to review.")

    def next_frame(self):
        if self.sample_position + 1 < len(self.sample_frames):
            self.sample_position += 1
            self.show_frame()

    def previous_frame(self):
        if self.sample_position > 0:
            self.sample_position -= 1
            self.show_frame()

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        point_ids = None
        if self.label_mode == "points":
            first_annotation = next(iter(self.labels.values()), None)
            if first_annotation:
                point_ids = ["center"] + [
                    f"feature_{index}"
                    for index in range(1, len(first_annotation["points"]))
                ]
        data = {
            "source_video": str(self.video_path),
            "fps": self.fps,
            "frame_size": [self.frame_width, self.frame_height],
            "sample_interval_seconds": (
                self.sample_frames[1] / self.fps if len(self.sample_frames) > 1 else None
            ),
            "label_type": (
                "tracked_points" if self.label_mode == "points" else "polygon_vertices"
            ),
            "annotations": [
                self.labels[frame_number]
                for frame_number in sorted(self.labels)
            ],
        }
        if point_ids is not None:
            data["point_ids"] = point_ids
        temporary_path = self.output_path.with_suffix(
            self.output_path.suffix + ".tmp"
        )
        with temporary_path.open("w", encoding="utf-8") as label_file:
            json.dump(data, label_file, indent=2)
            label_file.write("\n")
        temporary_path.replace(self.output_path)

    def close(self):
        self.save()
        self.capture.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Label target outlines or tracked points on sampled video frames."
    )
    parser.add_argument("--video", required=True, help="Challenge recording to label.")
    parser.add_argument(
        "--output",
        help="Path for the saved labels.",
    )
    parser.add_argument(
        "--mode",
        choices=("polygon", "points"),
        default="polygon",
        help="Label polygon outlines or persistent point tracks (default: polygon).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between sampled frames (default: 0.5).",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive.")
    if args.output is None:
        args.output = (
            "assets/shape_training/shape_point_labels.json"
            if args.mode == "points"
            else "assets/shape_training/shape_labels.json"
        )
    PolygonLabeler(args.video, args.output, args.interval, args.mode).run()


if __name__ == "__main__":
    main()
