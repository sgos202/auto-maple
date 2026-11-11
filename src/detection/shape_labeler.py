"""Label circle centers and radii on sampled frames from a challenge recording."""

import argparse
import base64
import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np


class CircleLabeler:
    def __init__(self, video_path, output_path, interval):
        self.video_path = Path(video_path).resolve()
        self.output_path = Path(output_path).resolve()
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
        self.pending_center = None
        self.display_scale = 1.0
        self.frame = None

        self.root = tk.Tk()
        self.root.title("Circle challenge labeler")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress-n>", lambda _event: self.next_frame())
        self.root.bind("<KeyPress-p>", lambda _event: self.previous_frame())
        self.root.bind("<Control-s>", lambda _event: self.save())

        instructions = tk.Label(
            self.root,
            text="Click the target center, then its edge. Right-click to reset. "
            "N: next  P: previous  Ctrl+S: save",
            anchor="w",
        )
        instructions.pack(fill=tk.X, padx=8, pady=(8, 4))

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
        tk.Button(controls, text="Save", command=self.save).pack(side=tk.RIGHT)

        self.show_frame()

    def _load_labels(self):
        if not self.output_path.exists():
            return {}

        with self.output_path.open("r", encoding="utf-8") as label_file:
            data = json.load(label_file)
        if data.get("source_video") != str(self.video_path):
            raise ValueError(
                f"Label file belongs to a different video: {self.output_path}"
            )
        if data.get("frame_size") != [self.frame_width, self.frame_height]:
            raise ValueError(f"Label file has different video dimensions: {self.output_path}")
        return {int(item["frame"]): item for item in data.get("annotations", [])}

    def _current_frame_number(self):
        return self.sample_frames[self.sample_position]

    def show_frame(self):
        frame_number = self._current_frame_number()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, self.frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"Could not read video frame {frame_number}.")

        max_width, max_height = 1100, 720
        self.display_scale = min(
            1.0,
            max_width / self.frame_width,
            max_height / self.frame_height,
        )
        display_width = round(self.frame_width * self.display_scale)
        display_height = round(self.frame_height * self.display_scale)
        display_frame = cv2.resize(
            self.frame, (display_width, display_height), interpolation=cv2.INTER_AREA
        )
        encoded, png = cv2.imencode(".png", display_frame)
        if not encoded:
            raise RuntimeError("Could not encode the video frame for display.")
        image_data = base64.b64encode(png.tobytes()).decode("ascii")
        self.photo = tk.PhotoImage(data=image_data, format="PNG")
        self.canvas.configure(width=display_width, height=display_height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.pending_center = None
        annotation = self.labels.get(frame_number)
        if annotation:
            self._draw_circle(annotation["center"], annotation["radius"])
        timestamp = frame_number / self.fps
        label_status = "labeled" if annotation else "not labeled"
        self.status.set(
            f"Frame {self.sample_position + 1}/{len(self.sample_frames)} "
            f"at {timestamp:.2f}s — {label_status} — "
            f"{len(self.labels)} labels saved to {self.output_path}"
        )

    def _draw_circle(self, center, radius):
        x, y = center
        x *= self.display_scale
        y *= self.display_scale
        radius *= self.display_scale
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            outline="#00ff00",
            width=2,
        )
        self.canvas.create_line(x - 5, y, x + 5, y, fill="#00ff00", width=2)
        self.canvas.create_line(x, y - 5, x, y + 5, fill="#00ff00", width=2)
        if self.pending_center is not None:
            pending_x, pending_y = self.pending_center
            self.canvas.create_line(
                pending_x * self.display_scale,
                pending_y * self.display_scale,
                x,
                y,
                fill="#00aaff",
                width=2,
            )

    def on_left_click(self, event):
        if not 0 <= event.x < self.canvas.winfo_width():
            return
        if not 0 <= event.y < self.canvas.winfo_height():
            return
        point = (event.x / self.display_scale, event.y / self.display_scale)
        if self.pending_center is None:
            self.pending_center = point
            self.canvas.delete("pending")
            self.canvas.create_oval(
                event.x - 4,
                event.y - 4,
                event.x + 4,
                event.y + 4,
                outline="#00aaff",
                width=2,
                tags="pending",
            )
            self.status.set("Now click the target edge to set its radius.")
            return

        radius = math.dist(self.pending_center, point)
        if radius < 2:
            self.status.set("Radius is too small; click a point on the circle edge.")
            return

        frame_number = self._current_frame_number()
        self.labels[frame_number] = {
            "frame": frame_number,
            "time_seconds": frame_number / self.fps,
            "center": [round(value, 2) for value in self.pending_center],
            "radius": round(radius, 2),
        }
        self.pending_center = None
        self.save()
        if self.sample_position + 1 < len(self.sample_frames):
            self.sample_position += 1
            self.show_frame()
        else:
            self.show_frame()
            self.status.set("End of video. Labels are saved; use Previous to review.")

    def on_right_click(self, _event):
        frame_number = self._current_frame_number()
        if self.pending_center is not None:
            self.pending_center = None
            self.canvas.delete("pending")
            self.status.set("Center selection cleared.")
        elif frame_number in self.labels:
            del self.labels[frame_number]
            self.save()
            self.show_frame()

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
        data = {
            "source_video": str(self.video_path),
            "fps": self.fps,
            "frame_size": [self.frame_width, self.frame_height],
            "sample_interval_seconds": (
                self.sample_frames[1] / self.fps if len(self.sample_frames) > 1 else None
            ),
            "label_type": "circle_center_radius",
            "annotations": [
                self.labels[frame_number]
                for frame_number in sorted(self.labels)
            ],
        }
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
        description="Label circle centers and radii on challenge video frames."
    )
    parser.add_argument("--video", required=True, help="Challenge recording to label.")
    parser.add_argument(
        "--output",
        default="assets/shape_training/circle_labels.json",
        help="Path for saved circle labels.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between sampled frames (default: 1).",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive.")
    CircleLabeler(args.video, args.output, args.interval).run()


if __name__ == "__main__":
    main()
