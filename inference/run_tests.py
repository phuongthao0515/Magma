"""
Step 3: Run model inference on test cases and report accuracy.

Reads the test cases JSON (filled in manually after annotate_screenshots.py),
runs the model on each screenshot+prompt, compares against expected output.

Usage:
    python /home/thaole/thao_le/Magma/inference/run_tests.py
"""

import gc
import json
import os
import re
import sys

import torch
from PIL import Image
from tqdm import tqdm

# ============ CONFIGURATION ============
PROJECT_ROOT = "/home/thaole/thao_le/Magma"
BASE_MODEL = "microsoft/Magma-8B"
# CHECKPOINT_PATH = "/home/thaole/thao_le/Magma/checkpoints/finetune-3apps-r32-a64-maxlen2560-focal-marks/checkpoint-3400"
CHECKPOINT_PATH="/home/thaole/thao_le/Magma/checkpoints/finetune-3apps-r32-a64-maxlen2560-focal-marks-5actions/checkpoint-3300"
IMG_SIZE = 768
TEST_CASES_JSONS = [
    ("word", "/home/thaole/thao_le/Magma/inference/tests/test_cases_word_v2.json"),
    ("excel", "/home/thaole/thao_le/Magma/inference/tests/test_cases_excel.json"),
    ("powerpoint", "/home/thaole/thao_le/Magma/inference/tests/test_cases_powerpoint.json"),
    ("terminate", "/home/thaole/thao_le/Magma/inference/tests/test_cases_terminate.json"),
]
RESULTS_DIR = "/home/thaole/thao_le/Magma/inference/tests/results"
RESULTS_FILENAME = "test_results_exp16_3apps_5actions_3300_beam2_with_terminate_273_testcases.json"

INSTRUCTION_TEMPLATE = (
    "Imagine that you are imitating humans doing GUI navigation step by step.\n\n"
    "You can perform actions such as CLICK, DOUBLE_CLICK, RIGHT_CLICK, MIDDLE_CLICK, "
    "MOVE, DRAG, SCROLL, HSCROLL, TYPE, PRESS, HOTKEY.\n\n"
    "Output format must be:\n"
    '{{\"ACTION\": action_type, \"MARK\": numeric_id, \"VALUE\": text_or_null}}\n\n'
    "Task: {task_prompt}\n\n"
    "Previous actions:\nNone\n\n"
    "For your convenience, UI elements are labeled with numeric marks.\n\n"
    "What is the next action?\n"
)
# =======================================

sys.path.insert(0, PROJECT_ROOT)

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig


def patch_pytorch():
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


def load_model():
    patch_pytorch()

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

    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.image_processor.base_img_size = IMG_SIZE

    return model, processor


def run_inference(model, processor, image, instruction):
    full_prompt = f"<image_start><image><image_end>\n{instruction}"
    convs = [
        {"role": "system", "content": "You are agent that can see, talk and act."},
        {"role": "user", "content": full_prompt},
    ]
    formatted_prompt = processor.tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
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
            num_beams=2,
            max_new_tokens=50,
            use_cache=True,
        )

    generate_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

    del inputs, output_ids, generate_ids
    gc.collect()
    torch.cuda.empty_cache()

    return response


def parse_action(text):
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except Exception:
            pass

    return {"raw_response": text, "parse_error": True}


def evaluate_sample(prediction, expected):
    if prediction.get("parse_error"):
        return {
            "action_match": False,
            "element_match": False,
            "value_match": False,
            "overall_match": False,
            "parse_error": True,
        }

    action_match = prediction.get("ACTION") == expected.get("ACTION")
    element_match = str(prediction.get("MARK")) == str(expected.get("MARK"))

    pred_value = str(prediction.get("VALUE", "")).strip()
    exp_value = str(expected.get("VALUE", "")).strip()
    value_match = pred_value == exp_value

    overall = action_match and element_match and value_match

    return {
        "action_match": action_match,
        "element_match": element_match,
        "value_match": value_match,
        "overall_match": overall,
        "parse_error": False,
    }


def load_samples_for_app(json_path):
    """Load samples for a single app's JSON. Returns list of (image, instruction, expected, prompt)."""
    with open(json_path) as f:
        test_cases = json.load(f)

    is_new_format = len(test_cases) > 0 and "conversations" in test_cases[0]
    samples = []
    if is_new_format:
        for sample in test_cases:
            instruction = sample["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "")
            expected = json.loads(sample["conversations"][1]["value"])
            samples.append((sample["image"], instruction, expected, instruction[:80]))
    else:
        for tc in test_cases:
            for prompt_info in tc["prompts"]:
                instruction = INSTRUCTION_TEMPLATE.format(task_prompt=prompt_info["prompt"])
                samples.append((tc["annotated_image"], instruction, prompt_info["expected"], prompt_info["prompt"]))
    return samples


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Pre-load sample lists per app (no model allocated yet)
    per_app_samples = []
    total_samples = 0
    for app, json_path in TEST_CASES_JSONS:
        samples = load_samples_for_app(json_path)
        per_app_samples.append((app, samples))
        total_samples += len(samples)
        print(f"Loaded {app}: {len(samples)} samples from {json_path}")
    print(f"\nTotal test samples: {total_samples}")

    all_results = []
    sample_idx = 0

    # Run each app with a fresh model load to avoid cross-app fragmentation (preserves num_beams=2 quality)
    for app_pos, (app, samples) in enumerate(per_app_samples):
        print(f"\n{'#'*80}")
        print(f"# APP {app_pos + 1}/{len(per_app_samples)}: {app.upper()}  ({len(samples)} samples)")
        print(f"{'#'*80}")

        print(f"Loading model: {BASE_MODEL} + {CHECKPOINT_PATH}")
        model, processor = load_model()
        print("Model loaded!")

        for image_path, instruction, expected, prompt in tqdm(samples, desc=f"{app}"):
            sample_idx += 1
            if not os.path.exists(image_path):
                print(f"  WARNING: image not found: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            raw_response = run_inference(model, processor, image, instruction)
            image.close()
            prediction = parse_action(raw_response)
            eval_result = evaluate_sample(prediction, expected)

            result = {
                "index": sample_idx,
                "app": app,
                "image": image_path,
                "prompt": prompt,
                "expected": expected,
                "prediction": prediction,
                "raw_response": raw_response,
                **eval_result,
            }
            all_results.append(result)

            status = "PASS" if eval_result["overall_match"] else "FAIL"
            print(f"  [{app}][{status}] {prompt[:50]}... -> {raw_response[:60]}")

        # Fully release model between apps to clear allocator fragmentation
        print(f"\nUnloading model after {app}...")
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    # Compute metrics
    total = len(all_results)
    if total == 0:
        print("No test cases found!")
        return

    metrics = {
        "total": total,
        "action_acc": sum(r["action_match"] for r in all_results) / total,
        "element_acc": sum(r["element_match"] for r in all_results) / total,
        "value_acc": sum(r["value_match"] for r in all_results) / total,
        "overall_acc": sum(r["overall_match"] for r in all_results) / total,
        "parse_error_rate": sum(r["parse_error"] for r in all_results) / total,
    }

    # Per-image breakdown
    print(f"\n{'='*80}")
    print(f"PER-IMAGE BREAKDOWN")
    print(f"{'='*80}")
    print(f"{'Image':<10} {'Total':>6} {'Action':>10} {'Element':>10} {'Value':>10} {'Overall':>10}")
    print(f"{'-'*80}")

    per_image = {}
    # Group results by image path
    from collections import defaultdict
    by_image = defaultdict(list)
    for r in all_results:
        by_image[r["image"]].append(r)

    for img_path, img_results in sorted(by_image.items()):
        folder = os.path.basename(img_path).replace(".png", "")[:10]
        n = len(img_results)
        act = sum(r["action_match"] for r in img_results)
        elem = sum(r["element_match"] for r in img_results)
        val = sum(r["value_match"] for r in img_results)
        ovr = sum(r["overall_match"] for r in img_results)
        per_image[folder] = {"total": n, "action": act, "element": elem, "value": val, "overall": ovr}
        print(f"{folder:<10} {n:>6} {f'{act}/{n}':>10} {f'{elem}/{n}':>10} {f'{val}/{n}':>10} {f'{ovr}/{n}':>10}")

    # Per-action-type breakdown
    print(f"\n{'='*80}")
    print(f"PER-ACTION-TYPE BREAKDOWN")
    print(f"{'='*80}")
    print(f"{'Action Type':<15} {'Count':>6} {'Act%':>7} {'Elem%':>7} {'Overall%':>9}")
    print(f"{'-'*80}")

    action_types = {}
    for r in all_results:
        atype = r["expected"].get("ACTION", "?")
        if atype not in action_types:
            action_types[atype] = {"total": 0, "action": 0, "element": 0, "overall": 0}
        action_types[atype]["total"] += 1
        action_types[atype]["action"] += r["action_match"]
        action_types[atype]["element"] += r["element_match"]
        action_types[atype]["overall"] += r["overall_match"]

    for atype in sorted(action_types.keys()):
        s = action_types[atype]
        n = s["total"]
        print(f"{atype:<15} {n:>6} {s['action']/n*100:>6.1f}% {s['element']/n*100:>6.1f}% {s['overall']/n*100:>8.1f}%")

    # Per-app pass/fail summary
    print(f"\n{'='*80}")
    print(f"PER-APP PASS/FAIL SUMMARY")
    print(f"{'='*80}")
    print(f"{'App':<15} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Pass %':>8} {'Fail %':>8}")
    print(f"{'-'*80}")

    per_app = {}
    by_app = defaultdict(list)
    for r in all_results:
        by_app[r["app"]].append(r)

    for app_name, app_results in sorted(by_app.items()):
        n = len(app_results)
        p = sum(r["overall_match"] for r in app_results)
        f = n - p
        per_app[app_name] = {"total": n, "pass": p, "fail": f, "pass_rate": p / n, "fail_rate": f / n}
        print(f"{app_name:<15} {n:>6} {p:>6} {f:>6} {p/n*100:>7.1f}% {f/n*100:>7.1f}%")

    # Print overall summary
    total_pass = sum(r["overall_match"] for r in all_results)
    total_fail = total - total_pass
    print(f"{'-'*80}")
    print(f"{'TOTAL':<15} {total:>6} {total_pass:>6} {total_fail:>6} {total_pass/total*100:>7.1f}% {total_fail/total*100:>7.1f}%")

    print(f"\n{'='*80}")
    print(f"OVERALL ({total} samples)")
    print(f"{'='*80}")
    print(f"  PASS: {total_pass}/{total}    FAIL: {total_fail}/{total}")
    print(f"  Action Accuracy:  {metrics['action_acc']*100:.1f}%")
    print(f"  Element Accuracy: {metrics['element_acc']*100:.1f}%")
    print(f"  Value Accuracy:   {metrics['value_acc']*100:.1f}%")
    print(f"  Overall Accuracy: {metrics['overall_acc']*100:.1f}%")
    print(f"  Parse Error Rate: {metrics['parse_error_rate']*100:.1f}%")
    print(f"{'='*80}")

    # Save detailed results
    output = {
        "checkpoint": CHECKPOINT_PATH,
        "base_model": BASE_MODEL,
        "img_size": IMG_SIZE,
        "test_cases_jsons": TEST_CASES_JSONS,
        "metrics": metrics,
        "per_app": per_app,
        "per_image": per_image,
        "per_action_type": action_types,
        "results": all_results,
    }

    results_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
