"""
Mind2Web Evaluation Script
Based on debug_eval_mind2web.ipynb.

Evaluates a fine-tuned Magma checkpoint (with LoRA adapters)
on the Mind2Web validation set.

Usage:
    python evaluation/eval_mind2web.py
"""

import os
import re
import json
import glob
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig


# ── Configuration (absolute paths, no CLI arguments) ─────────────────────────
PROJECT_ROOT = "/home/sonnguyen/thaole/magma/Magma"
BASE_MODEL = "microsoft/Magma-8B"

# Single checkpoint evaluation
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "finetune-mind2web-qlora", "checkpoint-1655")

# Multi-checkpoint evaluation: set EVAL_ALL = True to sweep all checkpoint-* dirs
EVAL_ALL = False
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "finetune-mind2web-qlora")

# Dataset
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "mind2web")
DATA_JSON = os.path.join(DATA_DIR, "mind2web_val.json")

# Output
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "mind2web_eval")
SAVE_IMAGES = False          # set True to write annotated debug PNGs
MAX_SAMPLES = None           # set an int to limit, or None for full eval
USE_4BIT = True
IMG_SIZE = 768               # larger image so mark IDs are more visible to the vision encoder


# ── PyTorch 2.10+ compatibility patch ────────────────────────────────────────
if not hasattr(torch, "_original_sum_backup"):
    torch._original_sum_backup = torch.sum

    def _patched_sum(input, *args, **kwargs):
        if isinstance(input, bool):
            input = torch.tensor(input, dtype=torch.long)
        elif (
            isinstance(input, torch.Tensor)
            and input.dtype == torch.bool
            and (len(args) > 0 or "dim" in kwargs)
        ):
            input = input.long()
        return torch._original_sum_backup(input, *args, **kwargs)

    torch.sum = _patched_sum


# ── Model loading (base model cached, LoRA adapters swapped) ─────────────────
_cached_base_model = None
_cached_processor = None


def load_base_model():
    """Load the base model + processor once and cache them."""
    global _cached_base_model, _cached_processor
    if _cached_base_model is not None:
        print("Using cached base model")
        return _cached_base_model, _cached_processor

    print(f"Loading base model: {BASE_MODEL}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=quantization_config,
        attn_implementation="eager",
    )

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.image_processor.base_img_size = IMG_SIZE

    _cached_base_model = model
    _cached_processor = processor
    return model, processor


def load_model(checkpoint_path: str):
    """Return base model with LoRA adapters attached + processor."""
    base_model, processor = load_base_model()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading LoRA adapters from: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        print("LoRA adapters loaded!")
    else:
        print(f"WARNING: checkpoint not found ({checkpoint_path}), using base model only")
        model = base_model

    model.eval()
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, processor, image: Image.Image, prompt: str) -> str:
    full_prompt = f"<image_start><image><image_end>\n{prompt}"
    convs = [
        {"role": "system", "content": "You are agent that can see, talk and act."},
        {"role": "user", "content": full_prompt},
    ]
    prompt_text = processor.tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(images=[image], texts=prompt_text, return_tensors="pt")
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
            use_cache=True,
        )

    generate_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    return processor.decode(generate_ids[0], skip_special_tokens=True).strip()


# ── Parsing & metrics ─────────────────────────────────────────────────────────
def parse_action(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {"raw_response": response, "parse_error": True}


def compute_metrics(results: list) -> dict:
    total = len(results)
    action_ok = elem_ok = value_ok = overall_ok = parse_err = 0
    type_total = 0
    type_value_ok = 0
    select_total = 0
    select_value_ok = 0

    for r in results:
        gt, pred = r["ground_truth"], r["prediction"]
        if pred.get("parse_error"):
            parse_err += 1
            continue

        a = pred.get("ACTION") == gt.get("ACTION")
        e = str(pred.get("MARK")) == str(gt.get("MARK"))
        v = str(pred.get("VALUE")) == str(gt.get("VALUE"))

        action_ok += int(a)
        elem_ok += int(e)
        value_ok += int(v)

        if gt.get("ACTION") == "TYPE":
            type_total += 1
            if v:
                type_value_ok += 1
        if gt.get("ACTION") == "SELECT":
            select_total += 1
            if v:
                select_value_ok += 1

        # Overall: all three must match
        if a and e and v:
            overall_ok += 1

    valid = total - parse_err
    return {
        "total_samples": total,
        "valid_samples": valid,
        "parse_errors": parse_err,
        "action_accuracy": action_ok / valid if valid else 0,
        "element_accuracy": elem_ok / valid if valid else 0,
        "value_accuracy": value_ok / valid if valid else 0,
        "overall_accuracy": overall_ok / valid if valid else 0,
        "type_value_accuracy": type_value_ok / type_total if type_total else 0,
        "type_total": type_total,
        "select_value_accuracy": select_value_ok / select_total if select_total else 0,
        "select_total": select_total,
    }


# ── Debug image annotation ────────────────────────────────────────────────────
def annotate_image(image, sample_id, gt, pred, raw_response):
    img = image.copy()
    w, h = img.size

    is_err = pred.get("parse_error", False)
    if is_err:
        status, color = "PARSE_ERROR", (200, 100, 0)
    else:
        a = pred.get("ACTION") == gt.get("ACTION")
        e = str(pred.get("MARK")) == str(gt.get("MARK"))
        v = str(pred.get("VALUE")) == str(gt.get("VALUE"))
        ok = a and e and v
        status = "CORRECT" if ok else "WRONG"
        color = (0, 150, 0) if ok else (200, 0, 0)

    lines = [
        f"[{sample_id}]  {status}",
        f"GT:   {json.dumps(gt)}",
        f"PRED: {json.dumps(pred)}" if not is_err else "PRED: [parse error]",
        f"RAW:  {raw_response[:120]}{'...' if len(raw_response) > 120 else ''}",
    ]

    font_size = max(14, min(20, w // 60))
    font_path = os.path.join(PROJECT_ROOT, "agents", "ui_agent", "util", "arial.ttf")
    try:
        font = ImageFont.truetype(font_path, font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    lh = font_size + 4
    banner_h = lh * len(lines) + 12
    out = Image.new("RGB", (w, h + banner_h), (255, 255, 255))
    out.paste(img, (0, banner_h))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, w, lh + 6], fill=color)
    y = 4
    for i, line in enumerate(lines):
        draw.text((8, y), line, fill=(255, 255, 255) if i == 0 else (0, 0, 0), font=font)
        y += lh
    return out


# ── Per-checkpoint evaluation ─────────────────────────────────────────────────
def evaluate_checkpoint(checkpoint_path: str, dataset: list):
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path}")
    print("=" * 60)

    model, processor = load_model(checkpoint_path)
    samples = dataset[: min(MAX_SAMPLES, len(dataset))] if MAX_SAMPLES else dataset

    ckpt_image_dir = None
    if SAVE_IMAGES:
        ckpt_name = os.path.basename(checkpoint_path)
        ckpt_image_dir = os.path.join(OUTPUT_DIR, "images", ckpt_name)
        os.makedirs(ckpt_image_dir, exist_ok=True)

    results = []
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        image_path = os.path.join(DATA_DIR, sample["image"])
        image = Image.open(image_path).convert("RGB")

        user_prompt = sample["conversations"][0]["value"].replace("<image>", "").strip()
        gt_raw = sample["conversations"][1]["value"]
        gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw

        response = run_inference(model, processor, image, user_prompt)
        pred = parse_action(response)

        results.append({
            "id": sample.get("id", str(idx)),
            "ground_truth": gt,
            "prediction": pred,
            "raw_response": response,
        })

        if ckpt_image_dir:
            ann = annotate_image(image, sample.get("id", str(idx)), gt, pred, response)
            is_err = pred.get("parse_error", False)
            if is_err:
                prefix = "ERR"
            else:
                a = pred.get("ACTION") == gt.get("ACTION")
                e = str(pred.get("MARK")) == str(gt.get("MARK"))
                v = str(pred.get("VALUE")) == str(gt.get("VALUE"))
                prefix = "OK" if (a and e and v) else "FAIL"
            ann.save(os.path.join(ckpt_image_dir, f"{idx:04d}_{prefix}_{sample.get('id', idx)}.png"))

    metrics = compute_metrics(results)
    metrics["checkpoint"] = checkpoint_path

    print(f"\nResults for {os.path.basename(checkpoint_path)}:")
    print(f"   Action Accuracy:    {metrics['action_accuracy']*100:.2f}%")
    print(f"   Element Accuracy:   {metrics['element_accuracy']*100:.2f}%")
    print(f"   Value Accuracy:     {metrics['value_accuracy']*100:.2f}%")
    print(f"   Overall Accuracy:   {metrics['overall_accuracy']*100:.2f}%")
    print(f"   Parse Errors:       {metrics['parse_errors']}/{metrics['total_samples']}")
    print(f"   TYPE Value Acc:     {metrics['type_value_accuracy']*100:.2f}% ({metrics['type_total']} samples)")
    print(f"   SELECT Value Acc:   {metrics['select_value_accuracy']*100:.2f}% ({metrics['select_total']} samples)")

    # Unload LoRA adapter but keep the cached base model
    if isinstance(model, PeftModel):
        model = model.unload()
    torch.cuda.empty_cache()
    return metrics, results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset from: {DATA_JSON}")
    with open(DATA_JSON) as f:
        dataset = json.load(f)
    print(f"Dataset size: {len(dataset)}")

    all_metrics = []

    if EVAL_ALL:
        checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*")))
        print(f"Found {len(checkpoints)} checkpoints")
        for ckpt in checkpoints:
            metrics, results = evaluate_checkpoint(ckpt, dataset)
            all_metrics.append(metrics)
            out = os.path.join(OUTPUT_DIR, f"{os.path.basename(ckpt)}_results.json")
            with open(out, "w") as f:
                json.dump({"metrics": metrics, "predictions": results}, f, indent=2)
    else:
        metrics, results = evaluate_checkpoint(CHECKPOINT_PATH, dataset)
        all_metrics.append(metrics)
        out = os.path.join(OUTPUT_DIR, f"{os.path.basename(CHECKPOINT_PATH)}_results.json")
        with open(out, "w") as f:
            json.dump({"metrics": metrics, "predictions": results}, f, indent=2)

    if len(all_metrics) > 1:
        print("\n" + "=" * 60)
        print("CHECKPOINT RANKING (by Overall Accuracy)")
        print("=" * 60)
        ranked = sorted(all_metrics, key=lambda x: x["overall_accuracy"], reverse=True)
        for i, m in enumerate(ranked):
            print(f"{i+1}. {os.path.basename(m['checkpoint'])}: {m['overall_accuracy']*100:.2f}%")
        print(f"\nBEST: {os.path.basename(ranked[0]['checkpoint'])} "
              f"({ranked[0]['overall_accuracy']*100:.2f}%)")
        with open(os.path.join(OUTPUT_DIR, "checkpoint_ranking.json"), "w") as f:
            json.dump(ranked, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
