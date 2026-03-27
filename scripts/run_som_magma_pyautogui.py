"""Run screenshot -> SoM -> Magma action prediction -> PyAutoGUI execution.

This script mirrors the notebook logic from:
- som.ipynb (OCR-only SoM mark generation)
- evaluation/debug_eval_mind2web.ipynb (prompt + inference flow)

It also reuses the same action execution semantics as scripts/test_pyautogui_exit.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import PeftModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

# OCR-only SoM path used in som.ipynb
import easyocr


DEFAULT_CHECKPOINT_GDRIVE_URL = "https://drive.google.com/drive/folders/1Gpjetc17505OMABaueAtc7qFGol3C4FB?usp=sharing"
DEFAULT_CHECKPOINT_LOCAL_DIR = Path("./checkpoints/mind2web_adapter")
DEFAULT_OUTPUT_ROOT_DIR = Path("./outputs/run_som")


def ensure_checkpoint_path(checkpoint_path: str, checkpoint_gdrive_url: str) -> Path:
    """Ensure adapter files exist locally, downloading from Google Drive if needed."""
    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    adapter_model_path = checkpoint_dir / "adapter_model.safetensors"
    adapter_config_path = checkpoint_dir / "adapter_config.json"

    if adapter_model_path.exists() and adapter_config_path.exists():
        print(f"Using cached adapter files at: {checkpoint_dir}")
        return checkpoint_dir

    if not checkpoint_gdrive_url:
        raise RuntimeError(
            "Checkpoint files are missing and no checkpoint_gdrive_url was provided."
        )

    try:
        gdown: Any = importlib.import_module("gdown")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: gdown. Install with: pip install gdown"
        ) from exc

    print("Downloading checkpoint folder from Google Drive...")
    print(checkpoint_gdrive_url)
    gdown.download_folder(
        url=checkpoint_gdrive_url,
        output=str(checkpoint_dir),
        quiet=False,
        use_cookies=False,
    )

    if not (adapter_model_path.exists() and adapter_config_path.exists()):
        raise RuntimeError(
            "Download finished but adapter files were not found in checkpoint directory."
        )

    return checkpoint_dir


def patch_torch_sum_for_magma() -> None:
    """Patch torch.sum bool-tensor behavior for newer torch versions."""
    if hasattr(torch, "_original_sum_backup"):
        return

    torch._original_sum_backup = torch.sum

    def _patched_sum(input, *args, **kwargs):
        if isinstance(input, bool):
            input = torch.tensor(input, dtype=torch.long)
        elif isinstance(input, torch.Tensor) and input.dtype == torch.bool and (len(args) > 0 or "dim" in kwargs):
            input = input.long()
        return torch._original_sum_backup(input, *args, **kwargs)

    torch.sum = _patched_sum


def patch_magma_init_weights(base_model: str):
    """Patch model _init_weights to avoid normal_ on non-floating tensors."""
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

    try:
        model_class = AutoModelForCausalLM._model_mapping[type(config)]
    except KeyError:
        auto_map = getattr(config, "auto_map", {}) or {}
        model_ref = auto_map.get("AutoModelForCausalLM")
        if not model_ref:
            raise RuntimeError(
                "Could not resolve model class for config type "
                f"{type(config).__name__}. Missing auto_map[AutoModelForCausalLM]."
            )
        model_class = get_class_from_dynamic_module(model_ref, base_model)

    def _safe_init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            if module.class_embedding.data.is_floating_point():
                module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            weight = getattr(module, "weight", None)
            if weight is not None and weight.data.is_floating_point():
                weight.data.normal_(mean=0.0, std=std)
            bias = getattr(module, "bias", None)
            if bias is not None and bias.data.is_floating_point():
                bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            weight = getattr(module, "weight", None)
            if weight is not None and weight.data.is_floating_point():
                weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    weight.data[module.padding_idx].zero_()

    model_class._init_weights = _safe_init_weights
    return model_class


def load_model_and_processor(base_model: str, checkpoint_path: str):
    """Load Magma base model + LoRA adapter with notebook-equivalent settings."""
    patch_torch_sum_for_magma()

    model_class = patch_magma_init_weights(base_model)

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }

    if torch.cuda.is_available():
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["torch_dtype"] = torch.float32

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    try:
        model = model_class.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            **load_kwargs,
        )
    except Exception as exc:
        msg = str(exc)
        bnb_env_error = (
            "frozenset" in msg and "discard" in msg
        ) or "bitsandbytes" in msg.lower()
        if not bnb_env_error:
            raise

        print(
            "Warning: 4-bit bitsandbytes loading failed; falling back to non-quantized model load."
        )
        print(f"Reason: {exc}")
        model = model_class.from_pretrained(
            base_model,
            **load_kwargs,
        )

    model = PeftModel.from_pretrained(model, checkpoint_path)
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    return model, processor


def parse_action(text: str) -> Dict[str, Any]:
    """Parse model output into a single action dict with notebook fallback behavior."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        return {"raw_response": text, "parse_error": True}
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group())
                if isinstance(parsed_json, dict):
                    return parsed_json
            except Exception:
                pass
        return {"raw_response": text, "parse_error": True}


def parse_actions(text: str) -> List[Dict[str, Any]]:
    """Parse model output into one or more actions like test_pyautogui_exit.py."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            actions = [item for item in parsed if isinstance(item, dict)]
            if actions:
                return actions
        return [{"raw_response": text, "parse_error": True}]
    except json.JSONDecodeError:
        single = parse_action(text)
        return [single]


def run_som_ocr_only(image: Image.Image) -> Tuple[Image.Image, Dict[int, Tuple[int, int]], List[Tuple[float, float, float, float]]]:
    """Run OCR-only SoM logic and return mark->screen-point mapping."""
    from agents.ui_agent.util.som import MarkHelper, plot_boxes_with_marks

    image_np = np.array(image)
    width, height = image.size

    ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    ocr_results = ocr_reader.readtext(image_np, text_threshold=0.5)

    all_bboxes_normalized: List[Tuple[float, float, float, float]] = []
    mark_to_point: Dict[int, Tuple[int, int]] = {}

    for i, detection in enumerate(ocr_results):
        coords, _text, _confidence = detection
        x1, y1 = coords[0]
        x2, y2 = coords[2]

        y_norm = y1 / height
        x_norm = x1 / width
        h_norm = (y2 - y1) / height
        w_norm = (x2 - x1) / width
        all_bboxes_normalized.append((y_norm, x_norm, h_norm, w_norm))

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        mark_to_point[i] = (cx, cy)

    if all_bboxes_normalized:
        mark_helper = MarkHelper()
        annotated = plot_boxes_with_marks(
            image.copy(),
            all_bboxes_normalized,
            mark_helper,
            edgecolor=(255, 0, 0),
            linewidth=2,
            normalized_to_pixel=True,
            add_mark=True,
        )
        return annotated, mark_to_point, all_bboxes_normalized

    return image.copy(), mark_to_point, all_bboxes_normalized


def run_inference(model, processor, image: Image.Image, instruction: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Run model inference with the exact prompt/generation logic from debug notebook."""
    full_prompt = f"<image_start><image><image_end>\n{instruction}"
    convs = [
        {"role": "system", "content": "You are agent that can see, talk and act."},
        {"role": "user", "content": full_prompt},
    ]

    formatted_prompt = processor.tokenizer.apply_chat_template(
        convs,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(images=[image], texts=formatted_prompt, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
    inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
    inputs = inputs.to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=256,
            use_cache=False,
        )

    generate_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
    predictions = parse_actions(response)
    return response, predictions


def execute_action(pyautogui, action_item: Dict[str, Any], mark_coords: Dict[int, Tuple[int, int]]) -> int:
    """Execute CLICK/TYPE using the same logic as scripts/test_pyautogui_exit.py."""
    action = str(action_item.get("ACTION", "")).upper()
    mark = action_item.get("MARK")
    value = action_item.get("VALUE")

    if action not in {"CLICK", "TYPE"}:
        print(f"Unsupported ACTION: {action}")
        return 2

    if mark is None:
        print("Invalid MARK id: None")
        return 2

    if not isinstance(mark, int):
        try:
            mark = int(mark)
        except (TypeError, ValueError):
            print(f"Invalid MARK id: {mark}")
            return 2

    if mark not in mark_coords:
        print(f"Unknown MARK id: {mark}")
        return 2

    x, y = mark_coords[mark]
    pyautogui.moveTo(x, y, duration=0.2)
    if action == "CLICK":
        pyautogui.click()
        time.sleep(2.0)

    if action == "TYPE":
        if value in (None, "None", ""):
            print("TYPE action requires VALUE")
            return 2
        pyautogui.write(str(value), interval=0.02)
    elif value not in (None, "None", ""):
        print("VALUE is ignored for CLICK action")

    print(f"Executed {action} on MARK {mark} at ({x}, {y})")
    return 0


def capture_image_and_offset(image_path: Optional[str]) -> Tuple[Image.Image, Tuple[int, int], bool]:
    """Capture screen when image_path is empty, otherwise load image from disk."""
    if image_path:
        image = Image.open(image_path).convert("RGB")
        return image, (0, 0), False

    try:
        pyautogui: Any = importlib.import_module("pyautogui")
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyautogui is required for screenshot capture mode") from exc

    screenshot = pyautogui.screenshot()
    return screenshot.convert("RGB"), (0, 0), True


def add_screen_offset(
    mark_coords: Dict[int, Tuple[int, int]],
    offset: Tuple[int, int],
) -> Dict[int, Tuple[int, int]]:
    """Translate image-relative points to screen points."""
    ox, oy = offset
    return {mark: (x + ox, y + oy) for mark, (x, y) in mark_coords.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture image, run SoM+Magma, and execute predicted action with PyAutoGUI."
    )
    parser.add_argument("--base_model", type=str, default="microsoft/Magma-8B")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(DEFAULT_CHECKPOINT_LOCAL_DIR),
        help=(
            "Path to LoRA adapter folder containing adapter_config.json and "
            "adapter_model.safetensors. If missing, it will be downloaded."
        ),
    )
    parser.add_argument(
        "--checkpoint_gdrive_url",
        type=str,
        default=DEFAULT_CHECKPOINT_GDRIVE_URL,
        help="Google Drive folder URL for adapter download fallback.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Mind2Web-style instruction text (without image token)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="Optional image path. If omitted, script captures current screen.",
    )
    parser.add_argument(
        "--save_image_path",
        type=str,
        default="",
        help=(
            "Optional path to save the input image. "
            "If omitted, saves to outputs/run_som/<timestamp>/input_image.png"
        ),
    )
    parser.add_argument(
        "--save_annotated_path",
        type=str,
        default="",
        help=(
            "Optional path to save SoM-annotated image. "
            "If omitted, saves to outputs/run_som/<timestamp>/som_annotated.png"
        ),
    )
    parser.add_argument(
        "--save_model_output_path",
        type=str,
        default="",
        help=(
            "Optional path to save model response/actions as JSON. "
            "If omitted, saves to outputs/run_som/<timestamp>/model_output.json"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run inference only and skip real PyAutoGUI action execution.",
    )
    parser.add_argument(
        "--start_delay",
        type=float,
        default=3.0,
        help="Seconds to wait before executing PyAutoGUI action.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        checkpoint_path = ensure_checkpoint_path(args.checkpoint_path, args.checkpoint_gdrive_url)
    except Exception as exc:
        print(f"Failed to prepare checkpoint: {exc}")
        return 2

    image_path = args.image_path.strip() or None

    run_dir = DEFAULT_OUTPUT_ROOT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    save_image_path = Path(args.save_image_path).expanduser() if args.save_image_path else run_dir / "input_image.png"
    save_annotated_path = (
        Path(args.save_annotated_path).expanduser()
        if args.save_annotated_path
        else run_dir / "som_annotated.png"
    )
    save_model_output_path = (
        Path(args.save_model_output_path).expanduser()
        if args.save_model_output_path
        else run_dir / "model_output.json"
    )

    print("Step 1/5: Acquire image")
    image, image_offset, captured_screen = capture_image_and_offset(image_path)
    print(f"Image size: {image.size}")

    save_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_image_path)
    print(f"Saved input image: {save_image_path}")

    print("Step 2/5: Run OCR-only SoM")
    som_annotated_image, mark_coords, bboxes = run_som_ocr_only(image)
    print(f"Detected candidate marks: {len(mark_coords)}")

    save_annotated_path.parent.mkdir(parents=True, exist_ok=True)
    som_annotated_image.save(save_annotated_path)
    print(f"Saved annotated image: {save_annotated_path}")

    if not bboxes:
        print("No OCR regions detected. Cannot map MARK to screen coordinates.")
        return 2

    print("Step 3/5: Load model")
    model, processor = load_model_and_processor(args.base_model, str(checkpoint_path))

    print("Step 4/5: Run model inference")
    response, predictions = run_inference(model, processor, som_annotated_image, args.instruction)
    print("Model response:")
    print(response)
    print("Parsed actions:")
    print(json.dumps(predictions, indent=2))

    save_model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with save_model_output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "base_model": args.base_model,
                "checkpoint_path": str(checkpoint_path),
                "instruction": args.instruction,
                "input_image_path": str(save_image_path),
                "annotated_image_path": str(save_annotated_path),
                "raw_response": response,
                "parsed_actions": predictions,
            },
            f,
            indent=2,
        )
    print(f"Saved model output: {save_model_output_path}")

    del model
    torch.cuda.empty_cache()

    if predictions[0].get("parse_error"):
        print("Parse error: model did not output valid JSON action")
        return 2

    print("Step 5/5: Execute action with PyAutoGUI")
    if args.dry_run:
        print("Dry-run enabled, skipping action execution.")
        return 0

    if not captured_screen:
        print("Action execution is only enabled in screenshot capture mode.")
        return 2

    try:
        pyautogui: Any = importlib.import_module("pyautogui")
    except ModuleNotFoundError:
        print("Missing dependency: pyautogui")
        print("Install with: pip install pyautogui")
        return 2

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1

    # In screenshot mode the offset is (0,0) for full-screen capture.
    action_coords = add_screen_offset(mark_coords, image_offset)

    print(json.dumps(action_coords, indent=2))
    print(f"Starting action in {args.start_delay:.1f}s. Move mouse to top-left corner to abort.")
    time.sleep(args.start_delay)

    try:
        for idx, action_item in enumerate(predictions, start=1):
            if not isinstance(action_item, dict):
                print(f"Action #{idx} is not a JSON object")
                return 2
            rc = execute_action(pyautogui, action_item, action_coords)
            if rc != 0:
                return rc
            time.sleep(0.2)
        return 0
    except pyautogui.FailSafeException:
        print("Aborted by fail-safe (mouse moved to top-left corner).")
        return 130


if __name__ == "__main__":
    sys.exit(main())
