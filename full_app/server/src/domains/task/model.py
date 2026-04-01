"""
Magma-8B model + OmniParser YOLO loading and inference.
Singleton pattern – models are loaded once and reused across requests.
All parameters match the reference notebook.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path

import gdown
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from ultralytics import YOLO

from src.domains.task.som import build_som_candidates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching notebook exactly)
# ---------------------------------------------------------------------------
BASE_MODEL = "microsoft/Magma-8B"
MODEL_IMAGE_SIZE = 640

# Checkpoint – will be downloaded from HuggingFace or Google Drive on first run.
# Override via env var MAGMA_CHECKPOINT_DIR if you have a local copy.
import os

CHECKPOINT_LOCAL_DIR = Path(
    os.environ.get(
        "MAGMA_CHECKPOINT_DIR",
        str(Path(__file__).resolve().parents[3] / "checkpoints" / "mind2web_adapter"),
    )
)

# OmniParser YOLO weights
WEIGHTS_DIR = Path(
    os.environ.get(
        "OMNIPARSER_WEIGHTS_DIR",
        str(Path(__file__).resolve().parents[3] / "weights"),
    )
)
OMNIPARSER_MODEL_PATH = WEIGHTS_DIR / "icon_detect" / "model.pt"

# Google Drive folder containing the LoRA adapter checkpoint
CHECKPOINT_GDRIVE_URL = "https://drive.google.com/drive/folders/18RNkzvGCehTvi6J1vqV4hb8E9_fkmvXR?usp=drive_link"

INSTRUCTION_TEMPLATE = (
    "Imagine that you are imitating humans doing web navigation for a task step by step. "
    "At each stage, you can see the webpage like humans by a screenshot and know the previous "
    "actions before the current step decided by yourself through recorded history. You need to "
    "decide on the following action to take. You can click an element with the mouse, select an "
    "option, or type text with the keyboard. The output format should be a dictionary like: \n"
    '{"ACTION": "CLICK" or "TYPE" or "SELECT", "MARK": a numeric id, e.g., 5, '
    '"VALUE": a string value for the action if applicable, otherwise None}.\n'
    "You are asked to complete the following task: {task_prompt}\n"
    "For your convenience, I have labeled the candidates with numeric marks and bounding boxes "
    "on the screenshot. What is the next action you would take?"
)

# ---------------------------------------------------------------------------
# torch.sum bool-tensor compatibility patch (from notebook)
# ---------------------------------------------------------------------------
if not hasattr(torch, "_original_sum_backup"):
    torch._original_sum_backup = torch.sum

    def _patched_sum(input, *args, **kwargs):
        if isinstance(input, bool):
            input = torch.tensor(input, dtype=torch.long)
        elif isinstance(input, torch.Tensor) and input.dtype == torch.bool and (len(args) > 0 or "dim" in kwargs):
            input = input.long()
        return torch._original_sum_backup(input, *args, **kwargs)

    torch.sum = _patched_sum
    logger.info("Applied torch.sum bool-tensor compatibility patch")

# ---------------------------------------------------------------------------
# Safe weight init patch (from notebook)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Singleton holders
# ---------------------------------------------------------------------------
_magma_model = None
_magma_processor = None
_yolo_model = None
_device: str = "cpu"


# ---------------------------------------------------------------------------
# OmniParser YOLO loading
# ---------------------------------------------------------------------------
def _download_omniparser_weights() -> None:
    if OMNIPARSER_MODEL_PATH.exists():
        return
    logger.info("Downloading OmniParser-v2.0 weights from HuggingFace …")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="microsoft/OmniParser-v2.0", local_dir=str(WEIGHTS_DIR), local_dir_use_symlinks=False)


def load_yolo() -> YOLO:
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    _download_omniparser_weights()
    if not OMNIPARSER_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO weights not found at: {OMNIPARSER_MODEL_PATH}")
    logger.info(f"Loading YOLO model from {OMNIPARSER_MODEL_PATH}")
    _yolo_model = YOLO(str(OMNIPARSER_MODEL_PATH))
    return _yolo_model


# ---------------------------------------------------------------------------
# Magma model loading
# ---------------------------------------------------------------------------
def _has_valid_checkpoint(path: Path) -> bool:
    has_config = (path / "adapter_config.json").exists()
    has_weights = (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
    return has_config and has_weights


def _download_checkpoint_folder(url: str, output_dir: Path, retries: int = 3) -> None:
    """Download LoRA adapter checkpoint from Google Drive (matching notebook retry logic)."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Checkpoint download attempt {attempt}/{retries} (use_cookies=False)")
            gdown.download_folder(
                url=url,
                output=str(output_dir),
                quiet=False,
                use_cookies=False,
                remaining_ok=False,
            )
            return
        except Exception as exc:
            last_exc = exc
            logger.warning(f"Checkpoint download failed on attempt {attempt}: {exc}")
            if attempt < retries:
                time.sleep(5 * attempt)

    logger.info("Retrying checkpoint download with use_cookies=True …")
    try:
        gdown.download_folder(
            url=url,
            output=str(output_dir),
            quiet=False,
            use_cookies=True,
            remaining_ok=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Google Drive blocked automated folder download (rate-limit or anti-bot challenge). "
            "Wait and retry, or mirror the checkpoint outside Drive and update MAGMA_CHECKPOINT_DIR."
        ) from (last_exc or exc)


def _ensure_checkpoint() -> None:
    """Download the LoRA adapter from Google Drive if not already present."""
    if _has_valid_checkpoint(CHECKPOINT_LOCAL_DIR):
        logger.info(f"Using cached checkpoint in {CHECKPOINT_LOCAL_DIR}")
        return

    logger.info(f"Checkpoint not found at {CHECKPOINT_LOCAL_DIR}. Downloading from Google Drive …")
    CHECKPOINT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    _download_checkpoint_folder(CHECKPOINT_GDRIVE_URL, CHECKPOINT_LOCAL_DIR)

    if not _has_valid_checkpoint(CHECKPOINT_LOCAL_DIR):
        raise RuntimeError(
            f"Checkpoint in {CHECKPOINT_LOCAL_DIR} is incomplete after download. "
            "Expected adapter_config.json and adapter_model.* files."
        )


def load_magma():
    global _magma_model, _magma_processor, _device

    if _magma_model is not None:
        return _magma_model, _magma_processor

    # Download checkpoint from Drive if missing
    _ensure_checkpoint()

    logger.info(f"Loading Magma base model: {BASE_MODEL}")

    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if hasattr(config, "auto_map") and "AutoModelForCausalLM" in config.auto_map:
        model_class = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], BASE_MODEL)
    else:
        model_class = AutoModelForCausalLM._model_mapping[type(config)]

    # Patch init weights
    model_class._init_weights = _safe_init_weights

    if torch.cuda.is_available():
        _device = "cuda"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": {"": 0},
            "attn_implementation": "eager",
            "quantization_config": quantization_config,
        }
    else:
        _device = "cpu"
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
            "attn_implementation": "eager",
        }

    try:
        model = model_class.from_pretrained(BASE_MODEL, **load_kwargs)
    except Exception as exc:
        msg = str(exc)
        if "frozenset" in msg or "bitsandbytes" in msg.lower():
            logger.warning("4-bit load failed; retrying without quantization_config")
            fallback_kwargs = {
                "trust_remote_code": True,
                "attn_implementation": "eager",
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            }
            if torch.cuda.is_available():
                fallback_kwargs["device_map"] = {"": 0}
            model = model_class.from_pretrained(BASE_MODEL, **fallback_kwargs)
        else:
            raise

    if _device == "cpu":
        model.to("cpu")

    logger.info(f"Loading LoRA adapter from {CHECKPOINT_LOCAL_DIR}")
    model = PeftModel.from_pretrained(model, str(CHECKPOINT_LOCAL_DIR))

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "base_img_size"):
        processor.image_processor.base_img_size = MODEL_IMAGE_SIZE
        logger.info(f"Set processor.image_processor.base_img_size={MODEL_IMAGE_SIZE}")
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    _magma_model = model
    _magma_processor = processor
    logger.info("Magma model loaded successfully")
    return _magma_model, _magma_processor


# ---------------------------------------------------------------------------
# Warm-up: call at server startup
# ---------------------------------------------------------------------------
def warmup() -> None:
    load_yolo()
    load_magma()
    logger.info("All models warmed up and ready")


# ---------------------------------------------------------------------------
# Action parsing (from notebook)
# ---------------------------------------------------------------------------
def parse_action(text: str) -> dict:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except Exception:
            pass

    return {"raw_response": text, "parse_error": True}


# ---------------------------------------------------------------------------
# Inference: screenshot + prompt → predicted action with coordinates
# ---------------------------------------------------------------------------
def infer(image: Image.Image, task_prompt: str) -> dict:
    """
    Run full SoM + Magma inference pipeline.

    Returns dict with keys:
        action   – "CLICK" | "TYPE" | "SELECT"
        x, y     – pixel coordinates of the predicted target (or None)
        value    – text value for TYPE/SELECT (or None)
        mark_id  – numeric mark index predicted by model (or None)
        raw_response – raw model text output
    """
    model, processor = load_magma()
    yolo = load_yolo()

    # 1. SoM detection + annotation
    som_annotated_image, candidate_bboxes = build_som_candidates(image, yolo)

    # 2. Build prompt (matching notebook)
    instruction = INSTRUCTION_TEMPLATE.replace("{task_prompt}", task_prompt)
    full_prompt = f"<image_start><image><image_end>\n{instruction}"

    convs = [
        {"role": "system", "content": "You are agent that can see, talk and act."},
        {"role": "user", "content": full_prompt},
    ]
    formatted_prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)

    if hasattr(model, "config") and getattr(model.config, "mm_use_image_start_end", False):
        formatted_prompt = formatted_prompt.replace("<image>", "<image_start><image><image_end>")

    # 3. Tokenize + run inference
    inputs = processor(images=[som_annotated_image], texts=formatted_prompt, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
    inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)

    if _device == "cuda":
        inputs = inputs.to(torch.bfloat16).to("cuda")
    else:
        inputs = inputs.to("cpu")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True,
        )

    generate_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

    # 4. Parse action
    prediction = parse_action(response)
    predicted_action = str(prediction.get("ACTION", "")).upper()
    mark_id = prediction.get("MARK")
    value = prediction.get("VALUE")

    coordinate = None
    if mark_id is not None and mark_id in candidate_bboxes:
        x1, y1, x2, y2 = candidate_bboxes[mark_id]
        coordinate = {"x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)}

    return {
        "action": predicted_action,
        "x": coordinate["x"] if coordinate else None,
        "y": coordinate["y"] if coordinate else None,
        "value": value if value and str(value).lower() != "none" else None,
        "mark_id": mark_id,
        "raw_response": response,
    }
