"""Train and evaluate a small circle-segmentation prototype from labeled frames."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


INPUT_HEIGHT = 192
INPUT_WIDTH = 288


def _load_examples(video_path, labels_path):
    with labels_path.open("r", encoding="utf-8") as label_file:
        label_data = json.load(label_file)

    if Path(label_data["source_video"]).resolve() != video_path.resolve():
        raise ValueError("The label file does not belong to the supplied video.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if label_data["frame_size"] != [frame_width, frame_height]:
        capture.release()
        raise ValueError("Video dimensions do not match the label file.")

    scale_x = INPUT_WIDTH / frame_width
    scale_y = INPUT_HEIGHT / frame_height
    examples = []
    try:
        for annotation in label_data["annotations"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(annotation["frame"]))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read labeled frame {annotation['frame']}."
                )

            resized = cv2.resize(
                frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA
            )
            image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0

            center_x, center_y = annotation["center"]
            center = (round(center_x * scale_x), round(center_y * scale_y))
            radius = round(
                annotation["radius"] * (scale_x + scale_y) / 2
            )
            mask = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.uint8)
            cv2.circle(mask, center, radius, 1, thickness=-1)
            examples.append(
                {
                    "frame": int(annotation["frame"]),
                    "time_seconds": float(annotation["time_seconds"]),
                    "image": image,
                    "mask": mask.astype(np.float32)[..., np.newaxis],
                    "center": tuple(map(float, annotation["center"])),
                    "radius": float(annotation["radius"]),
                    "original_frame": frame,
                }
            )
    finally:
        capture.release()

    if len(examples) < 6:
        raise ValueError(
            "At least 6 labeled frames are required for a train/validation split."
        )
    return examples


def _build_model():
    inputs = tf.keras.Input((INPUT_HEIGHT, INPUT_WIDTH, 3))

    def conv_block(value, filters):
        value = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False
        )(value)
        value = tf.keras.layers.BatchNormalization()(value)
        value = tf.keras.layers.Activation("relu")(value)
        value = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False
        )(value)
        value = tf.keras.layers.BatchNormalization()(value)
        return tf.keras.layers.Activation("relu")(value)

    enc1 = conv_block(inputs, 16)
    enc2 = conv_block(tf.keras.layers.MaxPooling2D()(enc1), 32)
    enc3 = conv_block(tf.keras.layers.MaxPooling2D()(enc2), 64)
    enc4 = conv_block(tf.keras.layers.MaxPooling2D()(enc3), 96)
    value = conv_block(tf.keras.layers.MaxPooling2D()(enc4), 128)

    for skip, filters in ((enc4, 96), (enc3, 64), (enc2, 32), (enc1, 16)):
        value = tf.keras.layers.UpSampling2D(interpolation="bilinear")(value)
        value = tf.keras.layers.Concatenate()((value, skip))
        value = conv_block(value, filters)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(value)
    model = tf.keras.Model(inputs, outputs, name="circle_segmenter")

    def dice_bce_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        axes = (1, 2, 3)
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        total = tf.reduce_sum(y_true + y_pred, axis=axes)
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (total + 1.0)
        return tf.reduce_mean(bce) + tf.reduce_mean(dice_loss)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=dice_bce_loss,
    )
    return model


def _make_dataset(images, masks, training, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    if training:
        dataset = dataset.shuffle(len(images), reshuffle_each_iteration=True)

        def augment(image, mask):
            flip = tf.random.uniform(()) > 0.5
            image = tf.cond(
                flip,
                lambda: tf.image.flip_left_right(image),
                lambda: image,
            )
            mask = tf.cond(
                flip,
                lambda: tf.image.flip_left_right(mask),
                lambda: mask,
            )
            image = tf.image.random_brightness(image, max_delta=0.12)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            return tf.clip_by_value(image, 0.0, 1.0), mask

        dataset = dataset.map(augment, num_parallel_calls=1).repeat()
    return dataset.batch(batch_size).prefetch(1)


def _largest_prediction(probability, frame_width, frame_height, threshold):
    binary = (probability >= threshold).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    candidates = [
        index for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 5
    ]
    if not candidates:
        return None

    index = max(candidates, key=lambda item: stats[item, cv2.CC_STAT_AREA])
    component = np.uint8(labels == index) * 255
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    (center_x, center_y), radius = cv2.minEnclosingCircle(
        max(contours, key=cv2.contourArea)
    )
    center_x *= frame_width / INPUT_WIDTH
    center_y *= frame_height / INPUT_HEIGHT
    radius *= (frame_width / INPUT_WIDTH + frame_height / INPUT_HEIGHT) / 2
    return (center_x, center_y), radius, binary


def _calibrate_threshold(model, examples, training_indices):
    probabilities = {
        index: model.predict(
            examples[index]["image"][np.newaxis, ...], verbose=0
        )[0, ..., 0]
        for index in training_indices
    }
    best_threshold = 0.5
    best_iou = -1.0
    for threshold in np.arange(0.05, 0.501, 0.025):
        scores = []
        for index in training_indices:
            example = examples[index]
            frame_height, frame_width = example["original_frame"].shape[:2]
            result = _largest_prediction(
                probabilities[index], frame_width, frame_height, threshold
            )
            predicted = (
                result[2].astype(bool)
                if result is not None
                else np.zeros(example["mask"].shape[:2], dtype=bool)
            )
            truth = example["mask"][..., 0] > 0.5
            union = np.count_nonzero(truth | predicted)
            scores.append(
                np.count_nonzero(truth & predicted) / union if union else 1.0
            )
        mean_iou = float(np.mean(scores))
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_threshold = float(threshold)
    return best_threshold, best_iou


def _evaluate_and_preview(
    model, examples, validation_indices, preview_path, threshold
):
    metrics = []
    tile_width, tile_height = 350, 234
    tiles = []

    for index, example in enumerate(examples):
        probability = model.predict(
            example["image"][np.newaxis, ...], verbose=0
        )[0, ..., 0]
        frame_height, frame_width = example["original_frame"].shape[:2]
        result = _largest_prediction(
            probability, frame_width, frame_height, threshold
        )
        truth = example["mask"][..., 0] > 0.5
        predicted_mask = (
            result[2].astype(bool)
            if result is not None
            else np.zeros_like(truth)
        )
        intersection = np.count_nonzero(truth & predicted_mask)
        union = np.count_nonzero(truth | predicted_mask)
        iou = intersection / union if union else 1.0

        predicted_center = None
        center_error = None
        radius_error = None
        if result is not None:
            predicted_center, predicted_radius, _ = result
            center_error = math.dist(predicted_center, example["center"])
            radius_error = abs(predicted_radius - example["radius"])

        metrics.append(
            {
                "frame": example["frame"],
                "time_seconds": example["time_seconds"],
                "held_out": index in validation_indices,
                "iou": iou,
                "center_error_pixels": center_error,
                "radius_error_pixels": radius_error,
                "prediction_found": result is not None,
            }
        )

        image = example["original_frame"].copy()
        ground_truth_center = tuple(
            round(value) for value in example["center"]
        )
        cv2.circle(
            image, ground_truth_center, round(example["radius"]), (255, 255, 0), 2
        )
        cv2.drawMarker(
            image, ground_truth_center, (255, 255, 0), cv2.MARKER_CROSS, 16, 2
        )
        if result is not None:
            predicted_center_px = tuple(
                round(value) for value in predicted_center
            )
            cv2.circle(
                image, predicted_center_px, round(predicted_radius), (0, 0, 255), 2
            )
            cv2.drawMarker(
                image, predicted_center_px, (0, 0, 255), cv2.MARKER_CROSS, 16, 2
            )
        title = f"{example['time_seconds']:.1f}s IoU={iou:.2f}"
        if index in validation_indices:
            title += " HOLDOUT"
        cv2.putText(
            image, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
        )
        tiles.append(cv2.resize(image, (tile_width, tile_height)))

    columns = 3
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        row.extend([np.zeros_like(tiles[0])] * (columns - len(row)))
        rows.append(cv2.hconcat(row))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview_path), cv2.vconcat(rows)):
        raise RuntimeError(f"Could not save preview image: {preview_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train and hold out a small circle segmentation prototype."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")

    examples = _load_examples(args.video.resolve(), args.labels.resolve())
    validation_indices = set(range(4, len(examples), 5))
    if not validation_indices:
        validation_indices.add(len(examples) - 1)
    training_indices = [
        index for index in range(len(examples))
        if index not in validation_indices
    ]
    if len(training_indices) < 2:
        raise ValueError("Not enough training frames after the holdout split.")

    train_images = np.stack([examples[index]["image"] for index in training_indices])
    train_masks = np.stack([examples[index]["mask"] for index in training_indices])
    validation_images = np.stack(
        [examples[index]["image"] for index in sorted(validation_indices)]
    )
    validation_masks = np.stack(
        [examples[index]["mask"] for index in sorted(validation_indices)]
    )

    batch_size = min(4, len(training_indices))
    model = _build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]
    model.fit(
        _make_dataset(train_images, train_masks, True, batch_size),
        steps_per_epoch=math.ceil(len(training_indices) / batch_size),
        validation_data=(validation_images, validation_masks),
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "circle_segmenter.keras"
    model.save(model_path)
    threshold, training_iou = _calibrate_threshold(
        model, examples, training_indices
    )
    preview_path = args.output_dir / "circle_predictions.png"
    metrics = _evaluate_and_preview(
        model, examples, validation_indices, preview_path, threshold
    )

    held_out = [item for item in metrics if item["held_out"]]
    valid_centers = [
        item["center_error_pixels"]
        for item in held_out
        if item["center_error_pixels"] is not None
    ]
    summary = {
        "video": str(args.video.resolve()),
        "labels": len(examples),
        "training_frames": len(training_indices),
        "held_out_frames": len(validation_indices),
        "training_mean_iou_at_selected_threshold": training_iou,
        "held_out_mean_iou": float(np.mean([item["iou"] for item in held_out])),
        "prediction_threshold": threshold,
        "held_out_mean_center_error_pixels": (
            float(np.mean(valid_centers)) if valid_centers else None
        ),
        "per_frame": metrics,
        "model": str(model_path.resolve()),
        "preview": str(preview_path.resolve()),
    }
    metrics_path = args.output_dir / "circle_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Threshold selected on training frames: {threshold:.3f} "
        f"(training mean IoU {training_iou:.3f})"
    )
    print(f"Held-out mean IoU: {summary['held_out_mean_iou']:.3f}")
    if summary["held_out_mean_center_error_pixels"] is not None:
        print(
            "Held-out mean center error: "
            f"{summary['held_out_mean_center_error_pixels']:.1f}px"
        )
    print(f"Model: {model_path.resolve()}")
    print(f"Preview: {preview_path.resolve()}")
    print(f"Metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
