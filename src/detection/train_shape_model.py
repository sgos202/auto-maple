"""Train a shared shape-segmentation prototype from circle and polygon labels."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.detection.train_circle_model import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    _build_model,
    _make_dataset,
)


def _load_examples(video_path, labels_path, run_name):
    with labels_path.open("r", encoding="utf-8") as label_file:
        label_data = json.load(label_file)

    if Path(label_data["source_video"]).resolve() != video_path.resolve():
        raise ValueError(f"{labels_path} does not belong to {video_path}.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if label_data["frame_size"] != [width, height]:
        capture.release()
        raise ValueError(f"Video dimensions do not match {labels_path}.")

    scale_x = INPUT_WIDTH / width
    scale_y = INPUT_HEIGHT / height
    examples = []
    try:
        for annotation in label_data["annotations"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(annotation["frame"]))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read labeled frame {annotation['frame']} from {video_path}."
                )

            if label_data["label_type"] == "circle_center_radius":
                center_x, center_y = annotation["center"]
                vertices = None
                center = (float(center_x), float(center_y))
                radius = float(annotation["radius"])
            elif label_data["label_type"] == "polygon_vertices":
                vertices = np.asarray(annotation["vertices"], dtype=np.float32)
                if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
                    raise ValueError(
                        f"Invalid polygon in frame {annotation['frame']} of {labels_path}."
                    )
                center = tuple(np.mean(vertices, axis=0).astype(float))
                radius = None
            else:
                raise ValueError(
                    f"Unsupported label type {label_data['label_type']!r} "
                    f"in {labels_path}."
                )

            resized = cv2.resize(
                frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA
            )
            image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            mask = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.uint8)
            if vertices is None:
                scaled_center = (
                    round(center[0] * scale_x),
                    round(center[1] * scale_y),
                )
                scaled_radius = round(radius * (scale_x + scale_y) / 2)
                cv2.circle(mask, scaled_center, scaled_radius, 1, thickness=-1)
            else:
                scaled_vertices = np.round(
                    vertices * np.array((scale_x, scale_y))
                ).astype(np.int32)
                cv2.fillPoly(mask, [scaled_vertices], 1)

            examples.append(
                {
                    "run": run_name,
                    "frame": int(annotation["frame"]),
                    "time_seconds": float(annotation["time_seconds"]),
                    "image": image,
                    "mask": mask.astype(np.float32)[..., np.newaxis],
                    "center": center,
                    "radius": radius,
                    "original_frame": frame,
                }
            )
    finally:
        capture.release()

    if len(examples) < 6:
        raise ValueError(f"At least 6 labeled frames are required in {labels_path}.")
    return examples


def _largest_component(probability, threshold):
    binary = (probability >= threshold).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    candidates = [
        index for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 5
    ]
    if not candidates:
        return None
    index = max(candidates, key=lambda item: stats[item, cv2.CC_STAT_AREA])
    return labels == index, centers[index]


def _mean_iou(examples, indices, probabilities, threshold):
    scores = []
    for index in indices:
        component = _largest_component(probabilities[index], threshold)
        predicted = (
            component[0]
            if component is not None
            else np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=bool)
        )
        truth = examples[index]["mask"][..., 0] > 0.5
        union = np.count_nonzero(truth | predicted)
        scores.append(
            np.count_nonzero(truth & predicted) / union if union else 1.0
        )
    return float(np.mean(scores))


def _select_threshold(examples, train_indices, train_probabilities):
    best_threshold = 0.5
    best_iou = -1.0
    for threshold in np.arange(0.05, 0.501, 0.025):
        score = _mean_iou(
            examples, train_indices, train_probabilities, float(threshold)
        )
        if score > best_iou:
            best_iou = score
            best_threshold = float(threshold)
    return best_threshold, best_iou


def _make_preview(examples, validation_indices, probabilities, threshold, path):
    metrics = []
    tiles = []
    for index, example in enumerate(examples):
        result = _largest_component(probabilities[index], threshold)
        truth = example["mask"][..., 0] > 0.5
        predicted = (
            result[0]
            if result is not None
            else np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=bool)
        )
        intersection = np.count_nonzero(truth & predicted)
        union = np.count_nonzero(truth | predicted)
        iou = intersection / union if union else 1.0
        center_error = None
        if result is not None:
            pred_x, pred_y = result[1]
            height, width = example["original_frame"].shape[:2]
            pred_center = (pred_x * width / INPUT_WIDTH, pred_y * height / INPUT_HEIGHT)
            center_error = math.dist(pred_center, example["center"])

        held_out = index in validation_indices
        metrics.append(
            {
                "run": example["run"],
                "frame": example["frame"],
                "time_seconds": example["time_seconds"],
                "held_out": held_out,
                "iou": iou,
                "center_error_pixels": center_error,
                "prediction_found": result is not None,
            }
        )

        image = example["original_frame"].copy()
        height, width = image.shape[:2]
        predicted_full = cv2.resize(
            predicted.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        truth_full = cv2.resize(
            truth.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        truth_contours, _ = cv2.findContours(
            truth_full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        pred_contours, _ = cv2.findContours(
            predicted_full.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(image, truth_contours, -1, (255, 255, 0), 2)
        cv2.drawContours(image, pred_contours, -1, (0, 0, 255), 2)
        title = f"{example['run']} {example['time_seconds']:.1f}s IoU={iou:.2f}"
        if held_out:
            title += " HOLDOUT"
        cv2.putText(
            image, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )
        tiles.append(cv2.resize(image, (350, 234)))

    rows = []
    for start in range(0, len(tiles), 3):
        row = tiles[start:start + 3]
        row.extend([np.zeros_like(tiles[0])] * (3 - len(row)))
        rows.append(cv2.hconcat(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.vconcat(rows)):
        raise RuntimeError(f"Could not save prediction preview: {path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train a shape segmenter from circle and/or polygon labels."
    )
    parser.add_argument("--circle-video", type=Path)
    parser.add_argument("--circle-labels", type=Path)
    parser.add_argument("--triangle-video", type=Path)
    parser.add_argument("--triangle-labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")

    if bool(args.circle_video) != bool(args.circle_labels):
        parser.error("--circle-video and --circle-labels must be used together.")
    if bool(args.triangle_video) != bool(args.triangle_labels):
        parser.error("--triangle-video and --triangle-labels must be used together.")
    if not args.circle_video and not args.triangle_video:
        parser.error("Provide at least one labeled video.")

    examples = []
    if args.circle_video:
        examples.extend(
            _load_examples(
                args.circle_video.resolve(), args.circle_labels.resolve(), "circle"
            )
        )
    if args.triangle_video:
        examples.extend(
            _load_examples(
                args.triangle_video.resolve(),
                args.triangle_labels.resolve(),
                "triangle",
            )
        )

    validation_indices = set()
    train_indices = []
    for run_name in {example["run"] for example in examples}:
        run_indices = [
            index for index, example in enumerate(examples)
            if example["run"] == run_name
        ]
        validation_indices.update(run_indices[4::5])
    train_indices = [
        index for index in range(len(examples))
        if index not in validation_indices
    ]
    if len(validation_indices) < 2:
        raise ValueError("Not enough held-out frames for evaluation.")

    train_images = np.stack([examples[index]["image"] for index in train_indices])
    train_masks = np.stack([examples[index]["mask"] for index in train_indices])
    validation_images = np.stack(
        [examples[index]["image"] for index in sorted(validation_indices)]
    )
    validation_masks = np.stack(
        [examples[index]["mask"] for index in sorted(validation_indices)]
    )
    batch_size = min(4, len(train_indices))

    model = _build_model()
    model.fit(
        _make_dataset(train_images, train_masks, True, batch_size),
        steps_per_epoch=math.ceil(len(train_indices) / batch_size),
        validation_data=(validation_images, validation_masks),
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )
        ],
        shuffle=False,
        verbose=2,
    )

    probabilities = {
        index: model.predict(
            examples[index]["image"][np.newaxis, ...], verbose=0
        )[0, ..., 0]
        for index in range(len(examples))
    }
    threshold, train_iou = _select_threshold(
        examples, train_indices, probabilities
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "shape_segmenter.keras"
    model.save(model_path)
    preview_path = output_dir / "shape_predictions.png"
    metrics = _make_preview(
        examples, validation_indices, probabilities, threshold, preview_path
    )
    held_out = [item for item in metrics if item["held_out"]]
    summary = {
        "runs": sorted({example["run"] for example in examples}),
        "labels": len(examples),
        "training_frames": len(train_indices),
        "held_out_frames": len(validation_indices),
        "holdout_method": "every fifth labeled frame within each recording",
        "training_mean_iou_at_selected_threshold": train_iou,
        "held_out_mean_iou": float(np.mean([item["iou"] for item in held_out])),
        "prediction_threshold": threshold,
        "model": str(model_path),
        "preview": str(preview_path),
        "per_frame": metrics,
    }
    metrics_path = output_dir / "shape_metrics.json"
    metrics_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Frames: {len(train_indices)} train, {len(validation_indices)} held out "
        f"(split within each supplied recording)"
    )
    print(
        f"Training IoU: {train_iou:.3f}; held-out IoU: "
        f"{summary['held_out_mean_iou']:.3f}"
    )
    print(f"Model: {model_path}")
    print(f"Preview: {preview_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
