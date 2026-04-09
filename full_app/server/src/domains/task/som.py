"""
Self-contained Set-of-Marks (SoM) utilities for UI element annotation.

Aligned with training preprocessing (preprocess_office_som.py):
- YOLO + EasyOCR combined detection
- OCR-aware overlap removal
- Matching thresholds: BOX_THRESHOLD=0.25, OCR_TEXT_THRESHOLD=0.9
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Font path (bundled with server)
# ---------------------------------------------------------------------------
_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "arial.ttf"

# ---------------------------------------------------------------------------
# Detection constants — ALIGNED WITH TRAINING (preprocess_office_som.py)
# ---------------------------------------------------------------------------
BOX_THRESHOLD = 0.25          # was 0.1, training uses 0.25
MIN_BOX_AREA = 0.0
OVERLAP_IOU_THRESHOLD = 0.7
MAX_MARKS = 100

# OCR parameters — matching training
OCR_TEXT_THRESHOLD = 0.9
OCR_MAX_AREA_RATIO = 0.05     # filter OCR boxes > 5% of image area

# Mark coordinate injection — matching inject_mark_coords.py
GRID_SIZE = 100


# ---------------------------------------------------------------------------
# EasyOCR singleton (lazy-loaded to avoid startup cost if not needed)
# ---------------------------------------------------------------------------
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        logger.info("Loading EasyOCR reader (first call)...")
        _ocr_reader = easyocr.Reader(["en"], gpu=True)
        logger.info("EasyOCR reader loaded")
    return _ocr_reader


# ---------------------------------------------------------------------------
# MarkHelper – manages font sizing and mark dimensions
# ---------------------------------------------------------------------------
class MarkHelper:
    def __init__(self) -> None:
        self.markSize_dict: dict = {}
        self.font_dict: dict = {}
        self.min_font_size = 12
        self.max_font_size = 14
        self.max_font_proportion = 0.04

    def __get_markSize(self, text: str, image_height: int, image_width: int, font):
        im = Image.new("RGB", (image_width, image_height))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return height, width

    def _setup_new_font(self, image_height: int, image_width: int) -> None:
        key = f"{image_height}_{image_width}"
        fontsize = self.min_font_size

        font_file = _FONT_PATH
        if not font_file.exists():
            raise FileNotFoundError(f"Required font not found: {font_file}")

        font = ImageFont.truetype(str(font_file), fontsize)
        while min(self.__get_markSize("555", image_height, image_width, font)) < min(
            self.max_font_size, self.max_font_proportion * min(image_height, image_width)
        ):
            fontsize += 1
            font = ImageFont.truetype(str(font_file), fontsize)

        self.font_dict[key] = font
        self.markSize_dict[key] = {
            1: self.__get_markSize("5", image_height, image_width, font),
            2: self.__get_markSize("55", image_height, image_width, font),
            3: self.__get_markSize("555", image_height, image_width, font),
        }

    def get_font(self, image_height: int, image_width: int):
        key = f"{image_height}_{image_width}"
        if key not in self.font_dict:
            self._setup_new_font(image_height, image_width)
        return self.font_dict[key]

    def get_mark_size(self, text_str: str, image_height: int, image_width: int):
        key = f"{image_height}_{image_width}"
        if key not in self.markSize_dict:
            self._setup_new_font(image_height, image_width)
        largest_size = self.markSize_dict[key].get(3, None)
        text_h, text_w = self.markSize_dict[key].get(len(text_str), largest_size)
        return text_h, text_w


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _calculate_iou(box1, box2, return_area: bool = False):
    y1, x1, h1, w1 = box1
    y2, x2, h2, w2 = box2

    y_min = max(y1, y2)
    x_min = max(x1, x2)
    y_max = min(y1 + h1, y2 + h2)
    x_max = min(x1 + w1, x2 + w2)

    intersection_area = max(0, y_max - y_min) * max(0, x_max - x_min)
    box1_area = h1 * w1
    box2_area = h2 * w2
    iou = intersection_area / (min(box1_area, box2_area) + 0.0001)

    if return_area:
        return iou, intersection_area
    return iou


def _calculate_nearest_corner_distance(box1, box2):
    y1, x1, h1, w1 = box1
    y2, x2, h2, w2 = box2
    corners1 = np.array([[y1, x1], [y1, x1 + w1], [y1 + h1, x1], [y1 + h1, x1 + w1]])
    corners2 = np.array([[y2, x2], [y2, x2 + w2], [y2 + h2, x2], [y2 + h2, x2 + w2]])
    distances = np.linalg.norm(corners1[:, np.newaxis] - corners2, axis=2)
    return np.min(distances)


def _find_least_overlapping_corner(bbox, bboxes, drawn_boxes, text_size, image_size):
    y, x, h, w = bbox
    h_text, w_text = text_size
    image_height, image_width = image_size
    corners = [
        (y - h_text, x),
        (y - h_text, x + w - w_text),
        (y, x + w),
        (y + h - h_text, x + w),
        (y + h, x + w - w_text),
        (y + h, x),
        (y + h - h_text, x - w_text),
        (y, x - w_text),
    ]
    best_corner = corners[0]
    max_flag = float("inf")

    for corner in corners:
        corner_bbox = (corner[0], corner[1], h_text, w_text)
        if (
            corner[0] < 0
            or corner[1] < 0
            or corner[0] + h_text > image_height
            or corner[1] + w_text > image_width
        ):
            continue
        max_iou = -(image_width + image_height)
        for other_bbox in bboxes + drawn_boxes:
            if np.array_equal(bbox, other_bbox):
                continue
            iou = _calculate_iou(corner_bbox, other_bbox, return_area=True)[1]
            max_iou = max(
                max_iou, iou - 0.0001 * _calculate_nearest_corner_distance(corner_bbox, other_bbox)
            )
        if max_iou < max_flag:
            max_flag = max_iou
            best_corner = corner

    return best_corner


# ---------------------------------------------------------------------------
# plot_boxes_with_marks – draw bounding boxes + index marks on image
# ---------------------------------------------------------------------------
def plot_boxes_with_marks(
    image: Image.Image,
    bboxes,
    mark_helper: MarkHelper,
    linewidth: int = 2,
    alpha: int = 0,
    edgecolor=None,
    fn_save=None,
    normalized_to_pixel: bool = True,
    add_mark: bool = True,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    if normalized_to_pixel:
        bboxes = [
            (
                int(y * image_height),
                int(x * image_width),
                int(h * image_height),
                int(w * image_width),
            )
            for y, x, h, w in bboxes
        ]

    for box in bboxes:
        y, x, h, w = box
        draw.rectangle([x, y, x + w, y + h], outline=edgecolor, width=linewidth)

    drawn_boxes = []
    for idx, bbox in enumerate(bboxes):
        text = str(idx)
        text_h, text_w = mark_helper.get_mark_size(text, image_height, image_width)
        corner_y, corner_x = _find_least_overlapping_corner(
            bbox, bboxes, drawn_boxes, (text_h, text_w), (image_height, image_width)
        )
        text_box = (corner_y, corner_x, text_h, text_w)

        if add_mark:
            draw.rectangle([corner_x, corner_y, corner_x + text_w, corner_y + text_h], fill="red")
            font = mark_helper.get_font(image_height, image_width)
            draw.text((corner_x, corner_y), text, fill="white", font=font)

        drawn_boxes.append(np.array(text_box))

    if fn_save is not None:
        image.save(fn_save)
    return image


# ---------------------------------------------------------------------------
# OCR detection — ported from preprocess_office_som.py
# ---------------------------------------------------------------------------
def _run_ocr(image: Image.Image) -> list[dict]:
    """Run EasyOCR on image, return structured element list in pixel xyxy."""
    ocr_reader = _get_ocr_reader()
    image_np = np.array(image)
    width, height = image.size

    result = ocr_reader.readtext(
        image_np, paragraph=False, text_threshold=OCR_TEXT_THRESHOLD
    )

    ocr_elements = []
    for item in result:
        coord = item[0]  # corner format [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = item[1]
        all_x = [pt[0] for pt in coord]
        all_y = [pt[1] for pt in coord]
        x1, y1 = int(min(all_x)), int(min(all_y))
        x2, y2 = int(max(all_x)), int(max(all_y))

        if x2 <= x1 or y2 <= y1:
            continue

        center_y_norm = ((y1 + y2) / 2) / height
        if center_y_norm > 0.93:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area >= OCR_MAX_AREA_RATIO * width * height:
            continue

        ocr_elements.append({
            "type": "text",
            "bbox": [x1, y1, x2, y2],
            "content": text,
        })

    return ocr_elements


# ---------------------------------------------------------------------------
# OCR-aware overlap removal — ported from preprocess_office_som.py
# ---------------------------------------------------------------------------
def _box_area(box: list) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(box1: list, box2: list) -> int:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou_xyxy(box1: list, box2: list) -> float:
    inter = _intersection_area(box1, box2)
    union = _box_area(box1) + _box_area(box2) - inter + 1e-6
    return inter / union


def _is_inside(inner: list, outer: list, threshold: float = 0.80) -> bool:
    """Check if inner box is mostly inside outer box."""
    inter = _intersection_area(inner, outer)
    inner_area = _box_area(inner)
    if inner_area == 0:
        return False
    return inter / inner_area >= threshold


def _remove_overlap_yolo(yolo_elements: list[dict], iou_threshold: float) -> list[dict]:
    """Remove overlapping YOLO boxes, keeping the smaller one."""
    if not yolo_elements:
        return []
    filtered = []
    for i, elem_i in enumerate(yolo_elements):
        is_valid = True
        for j, elem_j in enumerate(yolo_elements):
            if i != j and _iou_xyxy(elem_i["bbox"], elem_j["bbox"]) > iou_threshold:
                if _box_area(elem_i["bbox"]) > _box_area(elem_j["bbox"]):
                    is_valid = False
                    break
        if is_valid:
            filtered.append(elem_i)
    return filtered


def _remove_overlap_ocr_aware(
    yolo_elements: list[dict], iou_threshold: float, ocr_elements: list[dict]
) -> list[dict]:
    """OCR-aware overlap removal — matching preprocess_office_som.py.

    - If OCR text is inside a YOLO icon: absorb text into the icon
    - If YOLO icon is inside OCR text: drop the icon
    - Otherwise: keep both
    """
    if not yolo_elements and not ocr_elements:
        return []

    output = list(ocr_elements)
    yolo_deduped = _remove_overlap_yolo(yolo_elements, iou_threshold)

    for yolo_elem in yolo_deduped:
        absorbed = False
        for i, ocr_elem in enumerate(output):
            if ocr_elem["type"] != "text":
                continue

            if _is_inside(ocr_elem["bbox"], yolo_elem["bbox"]):
                yolo_elem["content"] = ocr_elem["content"]
                output.pop(i)
                output.append(yolo_elem)
                absorbed = True
                break

            if _is_inside(yolo_elem["bbox"], ocr_elem["bbox"]):
                absorbed = True
                break

        if not absorbed:
            output.append(yolo_elem)

    return output


# ---------------------------------------------------------------------------
# YOLO detection — returns element dicts in pixel xyxy
# ---------------------------------------------------------------------------
def _detect_yolo_elements(image: Image.Image, yolo_model: YOLO) -> list[dict]:
    width, height = image.size
    result = yolo_model.predict(source=image, conf=BOX_THRESHOLD, iou=0.1, verbose=False)

    raw_boxes = result[0].boxes.xyxy
    if len(raw_boxes) == 0:
        return []

    elements = []
    for box in raw_boxes.tolist():
        x1, y1, x2, y2 = box
        center_y_norm = ((y1 + y2) / 2) / height
        if center_y_norm > 0.93:
            continue
        bh_norm = (y2 - y1) / height
        bw_norm = (x2 - x1) / width
        if bh_norm * bw_norm < MIN_BOX_AREA:
            continue

        elements.append({
            "type": "icon",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "content": None,
        })

    return elements


# ---------------------------------------------------------------------------
# Combined YOLO + OCR detection — matching training pipeline
# ---------------------------------------------------------------------------
def detect_ui_elements_with_ocr(
    image: Image.Image, yolo_model: YOLO
) -> tuple[list[tuple], list[dict]]:
    """Combined YOLO + OCR detection pipeline (matches preprocess_office_som.py).

    Returns:
        bboxes_norm: list of (y, x, h, w) normalized — for MarkHelper drawing
        filtered_elements: list of element dicts with bbox/content — for metadata
    """
    width, height = image.size

    ocr_elements = _run_ocr(image)
    yolo_elements = _detect_yolo_elements(image, yolo_model)

    filtered_elements = _remove_overlap_ocr_aware(
        yolo_elements, OVERLAP_IOU_THRESHOLD, ocr_elements
    )

    if not filtered_elements:
        return [], []

    bboxes_norm = []
    valid_elements = []
    for elem in filtered_elements:
        x1, y1, x2, y2 = elem["bbox"]
        if x2 <= x1 or y2 <= y1:
            continue
        bboxes_norm.append((
            y1 / height,
            x1 / width,
            (y2 - y1) / height,
            (x2 - x1) / width,
        ))
        valid_elements.append(elem)
    filtered_elements = valid_elements

    if len(bboxes_norm) > MAX_MARKS:
        bboxes_norm = bboxes_norm[:MAX_MARKS]
        filtered_elements = filtered_elements[:MAX_MARKS]

    return bboxes_norm, filtered_elements


# ---------------------------------------------------------------------------
# Mark coordinate text for prompt injection — matching inject_mark_coords.py
# ---------------------------------------------------------------------------
def _is_clean_ocr(text: str) -> bool:
    """Check if OCR text is clean enough to include in the prompt."""
    if not text or not text.strip():
        return False
    if len(text.strip()) < 2:
        return False
    if not text.isascii():
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?()/-@&+_=")
    allowed_count = sum(1 for c in text if c in allowed)
    if len(text) > 0 and allowed_count / len(text) < 0.85:
        return False
    if len(text.strip()) <= 4 and not any(c.isalpha() for c in text):
        return False
    harsh = set("#*{}[]<>~^|\\`")
    harsh_count = sum(1 for c in text if c in harsh)
    if len(text) > 0 and harsh_count / len(text) > 0.3:
        return False
    return True


def format_marks_text(
    bboxes_norm: list[tuple],
    elements: list[dict],
    grid_size: int = GRID_SIZE,
) -> str:
    """Format mark metadata into text for prompt injection.

    Uses Magma pretrained format: Mark X at [gx,gy]
    Optionally appends clean OCR text: Mark X at [gx,gy] "File"
    """
    entries = []
    for idx, ((y, x, h, w), elem) in enumerate(zip(bboxes_norm, elements)):
        norm_cx = x + w / 2
        norm_cy = y + h / 2
        gx = min(int(norm_cx * grid_size), grid_size - 1)
        gy = min(int(norm_cy * grid_size), grid_size - 1)

        entry = f"Mark {idx} at [{gx},{gy}]"

        ocr_text = elem.get("content")
        if ocr_text and _is_clean_ocr(ocr_text):
            clean = ocr_text.strip().replace('"', "").replace("'", "")
            entry += f' "{clean}"'

        entries.append(entry)

    return ". ".join(entries)


# ---------------------------------------------------------------------------
# build_som_candidates – full pipeline: YOLO+OCR → annotate → return map
# ---------------------------------------------------------------------------
def build_som_candidates(
    image: Image.Image, yolo_model: YOLO
) -> tuple[Image.Image, dict[int, tuple[int, int, int, int]], str]:
    """Full SoM pipeline aligned with training preprocessing.

    Returns:
        som_annotated_image: image with SoM marks drawn
        candidate_bboxes: {mark_id: (x1, y1, x2, y2)} pixel coordinates
        marks_text: formatted mark coordinates for prompt injection
    """
    width, height = image.size
    bboxes_norm, elements = detect_ui_elements_with_ocr(image, yolo_model)

    som_annotated_image = image.copy()
    if bboxes_norm:
        mark_helper = MarkHelper()
        mark_helper.min_font_size = 12
        mark_helper.max_font_size = 14
        som_annotated_image = plot_boxes_with_marks(
            som_annotated_image,
            bboxes_norm,
            mark_helper,
            edgecolor=(255, 0, 0),
            linewidth=1,
            normalized_to_pixel=True,
            add_mark=True,
        )

    candidate_bboxes: dict[int, tuple[int, int, int, int]] = {
        idx: (
            int(x_norm * width),
            int(y_norm * height),
            int((x_norm + w_norm) * width),
            int((y_norm + h_norm) * height),
        )
        for idx, (y_norm, x_norm, h_norm, w_norm) in enumerate(bboxes_norm)
    }

    marks_text = format_marks_text(bboxes_norm, elements)

    return som_annotated_image, candidate_bboxes, marks_text
