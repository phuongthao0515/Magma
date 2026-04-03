"""
Self-contained Set-of-Marks (SoM) utilities for UI element annotation.
Extracted from agents/ui_agent/util/som.py to keep the server self-contained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Font path (bundled with server)
# ---------------------------------------------------------------------------
_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "arial.ttf"

# ---------------------------------------------------------------------------
# SoM detection constants (matching the notebook)
# ---------------------------------------------------------------------------
BOX_THRESHOLD = 0.1
MIN_BOX_AREA = 0.0
OVERLAP_IOU_THRESHOLD = 0.7


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
# Overlap removal (matching notebook logic)
# ---------------------------------------------------------------------------
def remove_overlap(boxes: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def iou(box1, box2):
        inter = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - inter + 1e-6
        a1, a2 = box_area(box1), box_area(box2)
        ratio1 = inter / a1 if a1 > 0 else 0
        ratio2 = inter / a2 if a2 > 0 else 0
        return max(inter / union, ratio1, ratio2)

    boxes_list = boxes.tolist()
    filtered = []
    for i, box1 in enumerate(boxes_list):
        is_valid = True
        for j, box2 in enumerate(boxes_list):
            if i != j and iou(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid = False
                break
        if is_valid:
            filtered.append(box1)
    return torch.tensor(filtered) if filtered else torch.zeros(0, 4)


# ---------------------------------------------------------------------------
# UI element detection with YOLO (OmniParser)
# ---------------------------------------------------------------------------
def detect_ui_elements(image: Image.Image, yolo_model: YOLO) -> list[tuple]:
    width, height = image.size
    result = yolo_model.predict(source=image, conf=BOX_THRESHOLD, iou=0.1, verbose=False)

    raw_boxes = result[0].boxes.xyxy
    if len(raw_boxes) == 0:
        return []

    xyxy_norm = raw_boxes / torch.tensor([width, height, width, height], device=raw_boxes.device)
    xyxy_norm = remove_overlap(xyxy_norm, iou_threshold=OVERLAP_IOU_THRESHOLD)

    bboxes = []
    for box in xyxy_norm.tolist():
        x1, y1, x2, y2 = box
        bh, bw = y2 - y1, x2 - x1
        if bh * bw >= MIN_BOX_AREA:
            center_y = (y1 + y2) / 2
            if center_y <= 0.93:
                bboxes.append((y1, x1, bh, bw))
    return bboxes


# ---------------------------------------------------------------------------
# build_som_candidates – full pipeline: detect → annotate → return map
# ---------------------------------------------------------------------------
def build_som_candidates(
    image: Image.Image, yolo_model: YOLO
) -> tuple[Image.Image, dict[int, tuple[int, int, int, int]]]:
    width, height = image.size
    all_bboxes_normalized = detect_ui_elements(image, yolo_model)

    som_annotated_image = image.copy()
    if all_bboxes_normalized:
        mark_helper = MarkHelper()
        mark_helper.min_font_size = 12
        mark_helper.max_font_size = 14
        som_annotated_image = plot_boxes_with_marks(
            som_annotated_image,
            all_bboxes_normalized,
            mark_helper,
            edgecolor=(255, 0, 0),
            linewidth=2,
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
        for idx, (y_norm, x_norm, h_norm, w_norm) in enumerate(all_bboxes_normalized)
    }

    return som_annotated_image, candidate_bboxes
