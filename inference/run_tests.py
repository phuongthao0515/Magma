"""
Step 3: Run model inference on test cases and report accuracy.

Reads the test cases JSON (filled in manually after annotate_screenshots.py),
runs the model on each screenshot+prompt, compares against expected output.

Usage:
    python /home/thaole/thao_le/Magma/inference/run_tests.py
"""

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
CHECKPOINT_PATH = "/home/thaole/thao_le/Magma/checkpoints/finetune-word-som-reduced-100-r32-a64-lr5e5/checkpoint-800"
IMG_SIZE = 768
TEST_CASES_JSON = "/home/thaole/thao_le/Magma/inference/tests/test_cases.json"
RESULTS_DIR = "/home/thaole/thao_le/Magma/inference/tests/results"

INSTRUCTION_TEMPLATE = (
    "Imagine that you are imitating humans doing GUI navigation step by step.\n\n"
    "You can perform actions such as CLICK, DOUBLE_CLICK, RIGHT_CLICK, MIDDLE_CLICK, "
    "MOVE, DRAG, SCROLL, HSCROLL, TYPE, PRESS, HOTKEY.\n\n"
    "Output format must be:\n"
    '{{\"ACTION\": action_type, \"MARK\": numeric_id, \"VALUE\": text_or_null}}\n\n'
    "Task: {task_prompt}\n"
    "For your convenience, I have labeled the candidates with numeric marks and "
    "bounding boxes on the screenshot. What is the next action you would take?"
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
            num_beams=1,
            max_new_tokens=50,
            use_cache=True,
        )

    generate_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
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

    value_match = True
    if expected.get("ACTION") == "TYPE":
        value_match = prediction.get("VALUE") == expected.get("VALUE")

    overall = action_match and element_match and value_match

    return {
        "action_match": action_match,
        "element_match": element_match,
        "value_match": value_match,
        "overall_match": overall,
        "parse_error": False,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading test cases: {TEST_CASES_JSON}")
    with open(TEST_CASES_JSON) as f:
        test_cases = json.load(f)

    total_prompts = sum(len(tc["prompts"]) for tc in test_cases)
    print(f"Test cases: {len(test_cases)} images, {total_prompts} prompts total")

    print(f"\nLoading model: {BASE_MODEL} + {CHECKPOINT_PATH}")
    model, processor = load_model()
    print("Model loaded!")

    all_results = []
    sample_idx = 0

    for tc in tqdm(test_cases, desc="Images"):
        annotated_path = tc["annotated_image"]
        if not os.path.exists(annotated_path):
            print(f"  WARNING: annotated image not found: {annotated_path}")
            continue

        image = Image.open(annotated_path).convert("RGB")

        for prompt_info in tc["prompts"]:
            sample_idx += 1
            prompt = prompt_info["prompt"]
            expected = prompt_info["expected"]

            instruction = INSTRUCTION_TEMPLATE.format(task_prompt=prompt)
            raw_response = run_inference(model, processor, image, instruction)
            prediction = parse_action(raw_response)
            eval_result = evaluate_sample(prediction, expected)

            result = {
                "index": sample_idx,
                "image": tc["image"],
                "prompt": prompt,
                "expected": expected,
                "prediction": prediction,
                "raw_response": raw_response,
                **eval_result,
            }
            all_results.append(result)

            status = "PASS" if eval_result["overall_match"] else "FAIL"
            print(f"  [{status}] {prompt[:50]}... -> {raw_response[:60]}")

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

    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST RESULTS ({total} samples)")
    print(f"{'='*60}")
    print(f"  Action Accuracy:  {metrics['action_acc']*100:.1f}%")
    print(f"  Element Accuracy: {metrics['element_acc']*100:.1f}%")
    print(f"  Value Accuracy:   {metrics['value_acc']*100:.1f}%")
    print(f"  Overall Accuracy: {metrics['overall_acc']*100:.1f}%")
    print(f"  Parse Error Rate: {metrics['parse_error_rate']*100:.1f}%")
    print(f"{'='*60}")

    # Save detailed results
    output = {
        "checkpoint": CHECKPOINT_PATH,
        "base_model": BASE_MODEL,
        "img_size": IMG_SIZE,
        "test_cases_json": TEST_CASES_JSON,
        "metrics": metrics,
        "results": all_results,
    }

    results_path = os.path.join(RESULTS_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
