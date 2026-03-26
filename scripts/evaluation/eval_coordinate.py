"""
Evaluate Word coordinate-based checkpoints.

Uses distance-based matching for COORDINATE field instead of exact MARK match.
Thresholds: 1% (~19px), 2% (~38px), 5% (~96px) on 1920x1080.

Usage:
    python scripts/evaluation/eval_coordinate.py
"""

import os
import sys
import json
import re
import glob
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import wandb


# ============ CONFIGURATION ============
BASE_MODEL = "microsoft/Magma-8B"
CHECKPOINT_DIR = "/home/thaole/thao_le/Magma/checkpoints/finetune-word-coord-qlora/checkpoint-900"
VAL_JSON = "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_coord_val.json"
IMAGE_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/office_images"
RESULTS_DIR = "/home/thaole/thao_le/Magma/results_new/word_coord_eval"
MAX_SAMPLES = None
BATCH_SIZE = 1
INCLUDE_BASE = False
EVAL_HOURS = 3
REST_MINUTES = 30
COORD_THRESHOLDS = [0.01, 0.02, 0.05]  # 1%, 2%, 5%
# =======================================


def patch_pytorch():
    if not hasattr(torch, '_original_sum_backup'):
        torch._original_sum_backup = torch.sum

    def _patched_sum(input, *args, **kwargs):
        if isinstance(input, bool):
            input = torch.tensor(input, dtype=torch.long)
        elif isinstance(input, torch.Tensor) and input.dtype == torch.bool and (len(args) > 0 or 'dim' in kwargs):
            input = input.long()
        return torch._original_sum_backup(input, *args, **kwargs)

    torch.sum = _patched_sum


def load_base_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=quantization_config,
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.image_processor.base_img_size = 768
    return model, processor


def run_inference_batch(model, processor, images, instructions):
    formatted_prompts = []
    for instruction in instructions:
        full_prompt = f"<image_start><image><image_end>\n{instruction}"
        convs = [
            {"role": "system", "content": "You are agent that can see, talk and act."},
            {"role": "user", "content": full_prompt},
        ]
        formatted_prompts.append(
            processor.tokenizer.apply_chat_template(
                convs, tokenize=False, add_generation_prompt=True
            )
        )

    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    all_pixel_values = []
    all_image_sizes = []
    all_input_ids = []
    all_attention_masks = []

    for img, prompt in zip(images, formatted_prompts):
        inputs = processor(images=[img], texts=prompt, return_tensors="pt")
        all_pixel_values.append(inputs['pixel_values'])
        all_image_sizes.append(inputs['image_sizes'])
        all_input_ids.append(inputs['input_ids'].squeeze(0))
        all_attention_masks.append(inputs['attention_mask'].squeeze(0))

    max_len = max(ids.shape[0] for ids in all_input_ids)
    pad_token_id = processor.tokenizer.pad_token_id

    padded_input_ids = []
    padded_attention_masks = []
    for ids, mask in zip(all_input_ids, all_attention_masks):
        pad_len = max_len - ids.shape[0]
        padded_input_ids.append(torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids]))
        padded_attention_masks.append(torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask]))

    batch_inputs = {
        'input_ids': torch.stack(padded_input_ids).to('cuda'),
        'attention_mask': torch.stack(padded_attention_masks).to('cuda'),
        'pixel_values': torch.stack(all_pixel_values).to('cuda').to(torch.bfloat16),
        'image_sizes': torch.stack(all_image_sizes).to('cuda'),
    }

    model.generation_config.pad_token_id = pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **batch_inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=50,
            use_cache=True
        )

    responses = []
    for i in range(len(images)):
        generate_ids = output_ids[i, batch_inputs["input_ids"].shape[-1]:]
        response = processor.decode(generate_ids, skip_special_tokens=True).strip()
        responses.append(response)

    return responses


def parse_action(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except (json.JSONDecodeError, Exception):
                pass
        return {"raw_response": text, "parse_error": True}


def parse_coordinate(value):
    if value is None or value == "None" or value == "null":
        return None
    if isinstance(value, list) and len(value) == 2:
        try:
            return [float(value[0]), float(value[1])]
        except (ValueError, TypeError):
            return None
    if isinstance(value, str):
        floats = re.findall(r'-?\d+\.?\d*', value)
        if len(floats) >= 2:
            return [float(floats[0]), float(floats[1])]
    return None


def evaluate_sample(prediction, ground_truth):
    if prediction.get('parse_error'):
        result = {
            "action_match": False,
            "element_match": False,
            "value_match": False,
            "overall_match": False,
            "parse_error": True,
            "coord_distance": None,
        }
        for t in COORD_THRESHOLDS:
            result[f"coord_{t}"] = False
        return result

    action_match = prediction.get('ACTION') == ground_truth.get('ACTION')

    gt_coord = parse_coordinate(ground_truth.get('COORDINATE'))
    pred_coord = parse_coordinate(prediction.get('COORDINATE'))

    coord_distance = None
    if gt_coord is not None and pred_coord is not None:
        coord_distance = math.sqrt(
            (pred_coord[0] - gt_coord[0]) ** 2 +
            (pred_coord[1] - gt_coord[1]) ** 2
        )
        element_match = coord_distance < 0.02  # primary: 2%
    elif gt_coord is None and pred_coord is None:
        element_match = True
    else:
        element_match = False

    value_match = True
    if ground_truth.get('ACTION') == 'TYPE':
        value_match = prediction.get('VALUE') == ground_truth.get('VALUE')

    overall = action_match and element_match and value_match

    result = {
        "action_match": action_match,
        "element_match": element_match,
        "value_match": value_match,
        "overall_match": overall,
        "parse_error": False,
        "coord_distance": coord_distance,
    }
    for t in COORD_THRESHOLDS:
        result[f"coord_{t}"] = (coord_distance is not None and coord_distance < t)

    return result


def evaluate_checkpoint(model, processor, val_data, max_samples=None):
    samples = val_data[:max_samples] if max_samples else val_data
    results = []

    valid_samples = []
    for i, sample in enumerate(samples):
        image_path = os.path.join(IMAGE_DIR, sample['image'])
        instruction = sample['conversations'][0]['value'].replace('<image>\n', '').replace('<image>', '')
        gt_text = sample['conversations'][1]['value']

        try:
            ground_truth = json.loads(gt_text)
        except json.JSONDecodeError:
            continue

        if not os.path.exists(image_path):
            continue

        valid_samples.append({
            "index": i,
            "id": sample.get('id', f'sample_{i}'),
            "image_path": image_path,
            "image_key": sample['image'],
            "instruction": instruction,
            "ground_truth": ground_truth,
        })

    for batch_start in tqdm(range(0, len(valid_samples), BATCH_SIZE), desc="Evaluating"):
        batch = valid_samples[batch_start:batch_start + BATCH_SIZE]

        images = [Image.open(s['image_path']).convert('RGB') for s in batch]
        instructions = [s['instruction'] for s in batch]

        try:
            responses = run_inference_batch(model, processor, images, instructions)
        except Exception as e:
            print(f"  Warning: Batch inference failed: {e}")
            responses = [""] * len(batch)

        for sample_info, response in zip(batch, responses):
            prediction = parse_action(response)
            eval_result = evaluate_sample(prediction, sample_info['ground_truth'])

            results.append({
                "id": sample_info['id'],
                "image": sample_info['image_key'],
                "ground_truth": sample_info['ground_truth'],
                "prediction": prediction,
                "raw_response": response,
                **eval_result,
            })

    total = len(results)
    if total == 0:
        return {"total": 0}, results

    metrics = {
        "total": total,
        "action_acc": sum(r['action_match'] for r in results) / total,
        "element_acc": sum(r['element_match'] for r in results) / total,
        "value_acc": sum(r['value_match'] for r in results) / total,
        "overall_acc": sum(r['overall_match'] for r in results) / total,
        "parse_error_rate": sum(r['parse_error'] for r in results) / total,
    }

    # Coordinate-specific metrics
    coord_results = [r for r in results if r.get('coord_distance') is not None]
    if coord_results:
        for t in COORD_THRESHOLDS:
            metrics[f"coord_{t}_acc"] = sum(r[f"coord_{t}"] for r in results) / total
        distances = [r['coord_distance'] for r in coord_results]
        metrics["coord_mean_dist"] = sum(distances) / len(distances)
        metrics["coord_median_dist"] = sorted(distances)[len(distances) // 2]

    return metrics, results


def print_summary_table(all_metrics):
    print("\n" + "=" * 110)
    print(f"{'Checkpoint':<20} {'Total':>6} {'Action%':>8} {'Elem%':>7} {'Val%':>6} {'Over%':>7} {'1%':>6} {'2%':>6} {'5%':>6} {'MeanDist':>9}")
    print("=" * 110)

    sorted_metrics = sorted(all_metrics, key=lambda x: x.get('overall_acc', 0), reverse=True)

    for m in sorted_metrics:
        c1 = m.get('coord_0.01_acc', 0) * 100
        c2 = m.get('coord_0.02_acc', 0) * 100
        c5 = m.get('coord_0.05_acc', 0) * 100
        md = m.get('coord_mean_dist', 0)
        print(f"{m['checkpoint']:<20} {m['total']:>6} {m['action_acc']*100:>7.1f} {m['element_acc']*100:>6.1f} "
              f"{m['value_acc']*100:>5.1f} {m['overall_acc']*100:>6.1f} {c1:>5.1f} {c2:>5.1f} {c5:>5.1f} {md:>8.4f}")

    print("=" * 110)
    best = sorted_metrics[0]
    print(f"\nBest checkpoint: {best['checkpoint']} (Overall: {best['overall_acc']*100:.1f}%)")


def main():
    patch_pytorch()

    wandb.init(project="magma-word-coord", name="eval-coord-checkpoints", job_type="eval")

    print(f"Loading val data from: {VAL_JSON}")
    with open(VAL_JSON) as f:
        val_data = json.load(f)
    print(f"Val samples: {len(val_data)}")

    if os.path.basename(CHECKPOINT_DIR).startswith("checkpoint-"):
        checkpoints = [CHECKPOINT_DIR]
    else:
        pattern = os.path.join(CHECKPOINT_DIR, "checkpoint-*")
        checkpoints = sorted(glob.glob(pattern), key=lambda x: int(x.split("-")[-1]))
    print(f"Found {len(checkpoints)} checkpoints: {[os.path.basename(c) for c in checkpoints]}")

    if INCLUDE_BASE:
        checkpoints = [None] + checkpoints

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nLoading base model: {BASE_MODEL}")
    base_model, processor = load_base_model()
    print("Base model loaded!")

    all_metrics = []
    start_time = time.time()

    for ckpt_path in checkpoints:
        elapsed_hours = (time.time() - start_time) / 3600
        if EVAL_HOURS > 0 and elapsed_hours >= EVAL_HOURS:
            print(f"\nRan for {elapsed_hours:.1f}h. Resting for {REST_MINUTES}min...")
            summary_file = os.path.join(RESULTS_DIR, "summary.json")
            with open(summary_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            time.sleep(REST_MINUTES * 60)
            start_time = time.time()
            print(f"Resuming evaluation...")

        if ckpt_path is None:
            ckpt_name = "base_model"
        else:
            ckpt_name = os.path.basename(ckpt_path)

        result_file = os.path.join(RESULTS_DIR, f"{ckpt_name}.json")
        if os.path.exists(result_file):
            print(f"\nSkipping {ckpt_name} (already evaluated)")
            with open(result_file) as f:
                saved = json.load(f)
            all_metrics.append(saved["metrics"])
            continue

        if ckpt_path is None:
            print(f"\n{'='*60}")
            print(f"Evaluating: {ckpt_name} (no LoRA)")
            print(f"{'='*60}")
            model = base_model
            model.eval()
        else:
            print(f"\n{'='*60}")
            print(f"Evaluating: {ckpt_name}")
            print(f"{'='*60}")
            model = PeftModel.from_pretrained(base_model, ckpt_path)
            model.eval()

        metrics, results = evaluate_checkpoint(model, processor, val_data, max_samples=MAX_SAMPLES)
        metrics['checkpoint'] = ckpt_name
        all_metrics.append(metrics)

        print(f"\n{ckpt_name} Results:")
        print(f"  Action Accuracy:  {metrics['action_acc']*100:.1f}%")
        print(f"  Element Accuracy: {metrics['element_acc']*100:.1f}% (2% threshold)")
        for t in COORD_THRESHOLDS:
            key = f"coord_{t}_acc"
            if key in metrics:
                print(f"  Coord <{t*100:.0f}%:       {metrics[key]*100:.1f}%")
        if 'coord_mean_dist' in metrics:
            print(f"  Mean Distance:    {metrics['coord_mean_dist']:.4f}")
        print(f"  Value Accuracy:   {metrics['value_acc']*100:.1f}%")
        print(f"  Overall Accuracy: {metrics['overall_acc']*100:.1f}%")
        print(f"  Parse Error Rate: {metrics['parse_error_rate']*100:.1f}%")

        step = int(ckpt_name.split("-")[-1]) if ckpt_name != "base_model" else 0
        wandb.log({
            "eval/action_acc": metrics['action_acc'],
            "eval/element_acc": metrics['element_acc'],
            "eval/value_acc": metrics['value_acc'],
            "eval/overall_acc": metrics['overall_acc'],
            "eval/parse_error_rate": metrics['parse_error_rate'],
            **{f"eval/coord_{t}_acc": metrics.get(f"coord_{t}_acc", 0) for t in COORD_THRESHOLDS},
        }, step=step)

        with open(result_file, 'w') as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        print(f"  Results saved to: {result_file}")

        if ckpt_path is not None:
            del model
            torch.cuda.empty_cache()

    if len(all_metrics) > 1:
        print_summary_table(all_metrics)

    summary_file = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    wandb.finish()


if __name__ == "__main__":
    main()
