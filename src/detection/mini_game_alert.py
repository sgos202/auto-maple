"""Detect mini-game warning pop-ups (e.g. the Lie Detector and Violetta prompts)."""

import cv2
import numpy as np


class MiniGamePromptDetector:
    REFERENCE_WIDTH = 996
    SCALE_FACTORS = (0.85, 0.95, 1.0, 1.05, 1.15)
    MATCH_THRESHOLD = 0.78
    # Fractions of the frame (x0, y0, x1, y1) to search. Default: the right side,
    # where the Lie Detector notice appears.
    DEFAULT_SEARCH_REGION = (0.52, 0.10, 1.0, 1.0)

    def __init__(self, template, search_region=None, threshold=None, downscale=1.0):
        if template is None or template.size == 0:
            raise ValueError("Mini-game prompt template is empty.")
        if template.ndim == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if template.ndim != 2:
            raise ValueError("Mini-game prompt template must be a grayscale or BGR image.")
        self.template = template
        self.search_region = search_region or self.DEFAULT_SEARCH_REGION
        self.threshold = self.MATCH_THRESHOLD if threshold is None else threshold
        # Values < 1 match on a shrunken image for speed (fine for large, smooth pop-ups).
        self.downscale = downscale

    def _scales(self, width):
        """Scales relative to REFERENCE_WIDTH, plus the template's native size."""
        base_scale = width / self.REFERENCE_WIDTH
        scales = [base_scale * factor for factor in self.SCALE_FACTORS]
        scales += [factor for factor in self.SCALE_FACTORS if factor not in scales]
        return scales

    def detect(self, frame):
        if frame is None or frame.ndim not in (2, 3):
            raise ValueError("Screen frame must be a grayscale, BGR, or BGRA image.")
        height, width = frame.shape[:2]
        if frame.ndim == 2:
            gray = frame
        elif frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        elif frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported screen frame shape: {frame.shape}")

        fx0, fy0, fx1, fy1 = self.search_region
        search_area = gray[round(height * fy0):round(height * fy1),
                           round(width * fx0):round(width * fx1)]
        down = self.downscale
        if down != 1:
            search_area = cv2.resize(search_area, None, fx=down, fy=down,
                                     interpolation=cv2.INTER_AREA)
        for scale in self._scales(width):
            scale *= down
            template_width = max(1, round(self.template.shape[1] * scale))
            template_height = max(1, round(self.template.shape[0] * scale))
            if template_width > search_area.shape[1] or template_height > search_area.shape[0]:
                continue
            interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            template = cv2.resize(
                self.template,
                (template_width, template_height),
                interpolation=interpolation,
            )
            result = cv2.matchTemplate(
                search_area, template, cv2.TM_CCOEFF_NORMED
            )
            if cv2.minMaxLoc(result)[1] >= self.threshold:
                return True
        return False


def _to_gray(image):
    if image is None or image.ndim not in (2, 3):
        raise ValueError("Image must be a grayscale, BGR, or BGRA array.")
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def white_text_mask(image):
    """
    Bright, colourless pixels (white text) as a soft 0-255 mask; everything else
    black. For pop-ups drawn see-through over the game, where the picture behind
    the text changes but the text itself stays the same.
    """
    if image.ndim == 2:
        mask = cv2.inRange(image, 200, 255)
    else:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
    return cv2.GaussianBlur(mask, (3, 3), 0)    # tolerates a little scaling and blur


class PopUpAnchorDetector:
    """
    Detects a pop-up from small, distinctive pieces ("anchors") cropped from a
    reference screenshot, e.g. its title text or NPC portrait. The pop-up counts
    as visible when ANY anchor matches, so it still triggers if the real pop-up's
    body text, size, or surroundings differ from the reference.

    WHITE_TEXT: match only the white text in the anchors and the screen (for a
    see-through pop-up whose background is whatever the game shows behind it).
    """

    SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)
    MATCH_THRESHOLD = 0.8

    def __init__(self, anchors, threshold=None, scales=None, downscale=1.0, white_text=False):
        self.white_text = white_text
        self._prepare = white_text_mask if white_text else _to_gray
        self.anchors = []
        for anchor in anchors:
            if anchor is None or anchor.size == 0:
                raise ValueError("Pop-up anchor image is empty.")
            self.anchors.append(self._prepare(anchor))
        if not self.anchors:
            raise ValueError("At least one pop-up anchor is required.")
        self.threshold = self.MATCH_THRESHOLD if threshold is None else threshold
        self.scales = tuple(scales or self.SCALES)
        self.downscale = downscale
        # Pre-resize every anchor at every scale once instead of every frame.
        self._templates = []
        for anchor in self.anchors:
            for scale in self.scales:
                scale *= downscale
                size = (max(1, round(anchor.shape[1] * scale)),
                        max(1, round(anchor.shape[0] * scale)))
                interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
                self._templates.append(cv2.resize(anchor, size, interpolation=interpolation))

    def best_score(self, frame):
        """Returns the highest anchor match score in FRAME (0 to 1)."""
        if frame is None or frame.ndim not in (2, 3):
            raise ValueError("Image must be a grayscale, BGR, or BGRA array.")
        gray = self._prepare(frame)
        if self.downscale != 1:
            gray = cv2.resize(gray, None, fx=self.downscale, fy=self.downscale,
                              interpolation=cv2.INTER_AREA)
        best = 0.0
        for template in self._templates:
            if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                continue
            score = cv2.minMaxLoc(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED))[1]
            if score >= self.threshold:
                return score
            best = max(best, score)
        return best

    def detect(self, frame):
        return self.best_score(frame) >= self.threshold


class RisingEdgeAlert:
    """Trigger once per prompt appearance and re-arm after brief disappearance."""

    def __init__(self, release_frames=3):
        if release_frames < 1:
            raise ValueError("release_frames must be positive.")
        self.release_frames = release_frames
        self.active = False
        self.missing_frames = 0

    def update(self, detected):
        if detected:
            self.missing_frames = 0
            if not self.active:
                self.active = True
                return True
            return False

        if self.active:
            self.missing_frames += 1
            if self.missing_frames >= self.release_frames:
                self.active = False
                self.missing_frames = 0
        return False
