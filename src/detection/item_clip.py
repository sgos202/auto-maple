"""
Zero-shot "which of these pictures is a <name>?" for the item quiz, using CLIP
(an image/text model) run with onnxruntime. Used when the quiz asks for an item
that has no picture in assets/quiz_items yet ("Click a bat").

Optional: needs `pip install onnxruntime tokenizers` and these files from
huggingface.co/Xenova/clip-vit-base-patch32 in models/clip/:
    onnx/vision_model_quantized.onnx, onnx/text_model_quantized.onnx, tokenizer.json
Without them, available() is False and the quiz falls back to elimination.

Check a screenshot:  python -m src.detection.item_clip IMAGE NAME
"""

import os

import cv2
import numpy as np

MODEL_DIR = 'models/clip'
VISION = 'vision_model_quantized.onnx'
TEXT = 'text_model_quantized.onnx'
TOKENIZER = 'tokenizer.json'
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
PROMPT = 'a pixel art icon of a {}'
# Other things a quiz picture could be, so "bat" is judged against something.
OTHERS = ('popsicle', 'witch hat', 'mushroom', 'snail', 'slime', 'pig', 'tree', 'flower',
          'apple', 'sword', 'shield', 'potion', 'coin', 'star', 'hat', 'shoe', 'fish', 'cat',
          'dog', 'bird', 'ghost', 'pumpkin', 'cake', 'candy', 'leaf', 'key', 'book', 'ball')

_model = None
_failed = False


def available():
    return _load() is not None


def _load():
    global _model, _failed
    if _model is not None or _failed:
        return _model
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        paths = [os.path.join(MODEL_DIR, f) for f in (VISION, TEXT, TOKENIZER)]
        if not all(os.path.isfile(p) for p in paths):
            raise FileNotFoundError(f'CLIP model files missing in {MODEL_DIR}')
        options = ort.SessionOptions()
        options.log_severity_level = 3
        vision = ort.InferenceSession(paths[0], options, providers=['CPUExecutionProvider'])
        text = ort.InferenceSession(paths[1], options, providers=['CPUExecutionProvider'])
        _model = (vision, text, Tokenizer.from_file(paths[2]), {})
    except Exception as error:
        _failed = True
        print(f'[QUIZ] AI item matching is off ({error}).')
    return _model


def _normalise(v):
    return v / max(1e-8, float(np.linalg.norm(v)))


def _text(words):
    vision, text, tokenizer, cache = _model
    if words not in cache:
        ids = np.array([tokenizer.encode(PROMPT.format(words)).ids], np.int64)
        feeds = {}
        for spec in text.get_inputs():
            feeds[spec.name] = ids if spec.name == 'input_ids' else np.ones_like(ids)
        names = [o.name for o in text.get_outputs()]
        output = text.run(None, feeds)
        embeds = output[names.index('text_embeds')] if 'text_embeds' in names else output[0]
        cache[words] = _normalise(embeds.reshape(-1))
    return cache[words]


def _image(picture, mask=None):
    """CLIP embedding of a BGR picture; MASK (bool) puts it on a plain background."""
    vision = _model[0]
    image = picture[:, :, :3].astype(np.float32)
    # Black, not white: white items (the witch hat) vanish on white. On the
    # popsicle quiz black gave popsicle 0.83 / witch hat 0.29 vs 0.46 / 0.01 on white.
    if mask is not None:
        image[~mask] = 0.0
    h, w = image.shape[:2]
    side = int(max(h, w) * 1.15)                  # square with a margin, no cropping
    canvas = np.full((side, side, 3), 0.0 if mask is not None else float(image.mean()),
                     np.float32)
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y + h, x:x + w] = image
    canvas = cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    rgb = canvas[:, :, ::-1] / 255.0
    tensor = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    names = [o.name for o in vision.get_outputs()]
    output = vision.run(None, {vision.get_inputs()[0].name: tensor})
    embeds = output[names.index('image_embeds')] if 'image_embeds' in names else output[0]
    return _normalise(embeds.reshape(-1))


def rank(name, pictures):
    """For PICTURES [(bgr, mask)], how likely (0-1) each is a NAME, or None when
    the model isn't available. Each picture is judged against OTHERS."""
    if _load() is None:
        return None
    # (no rival that is part of the name or contains it: "hat" vs "witch hat")
    labels = [name] + [o for o in OTHERS if o not in name and name not in o]
    texts = np.stack([_text(label) for label in labels])
    result = []
    for picture, mask in pictures:
        logits = 100.0 * texts @ _image(picture, mask)
        p = np.exp(logits - logits.max())
        result.append(float(p[0] / p.sum()))
    return result


def alike(pictures):
    """How alike (cosine, -1..1) each pair of PICTURES [(bgr, mask)] looks to the
    model, as an NxN array, or None when the model isn't available. For the
    "Click the one that is different" quiz."""
    if _load() is None:
        return None
    embeds = np.stack([_image(picture, mask) for picture, mask in pictures])
    return embeds @ embeds.T


def main():
    import sys
    from src.detection import letter_quiz as lq
    if len(sys.argv) != 3:
        print(__doc__)
        return
    frame = cv2.imread(sys.argv[1])
    renderer = lq._TitleRenderer()
    title = lq.read_title(frame, renderer, items=lq.Items().names)
    if title is None:
        print('No quiz title found.')
        return
    found = lq.find_items(frame, title)
    scores = rank(sys.argv[2], [(frame[b[1]:b[3], b[0]:b[2]], m[b[1]:b[3], b[0]:b[2]])
                                for b, m in found])
    for (box, _), score in zip(found, scores or [None] * len(found)):
        print(box, 'n/a' if score is None else f'{sys.argv[2]} {score:.2f}')


if __name__ == '__main__':
    main()
