"""
Mind2Web Evaluation Script
Evaluate fine-tuned checkpoints on Mind2Web dataset and select the best one.

Usage:
    # Evaluate single checkpoint
    python scripts/evaluation/eval_mind2web.py \
        --checkpoint ./checkpoints/finetune-mind2web-qlora/checkpoint-500 \
        --data datasets/mind2web \
        --max_samples 100

    # Evaluate all checkpoints and find best
    python scripts/evaluation/eval_mind2web.py \
        --checkpoint_dir ./checkpoints/finetune-mind2web-qlora \
        --data datasets/mind2web \
        --eval_all
"""

import os
import json
import argparse
import glob
import torch
from tqdm import tqdm
from PIL import Image
from datasets import Dataset
from peft import PeftModel
from transformers import BitsAndBytesConfig

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from magma.processing_magma import MagmaProcessor
from magma.modeling_magma import MagmaForCausalLM


def load_model(checkpoint_path: str, base_model: str = "microsoft/Magma-8B", use_4bit: bool = True):
    """Load fine-tuned model with LoRA adapters."""
    print(f"Loading base model: {base_model}")

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = MagmaForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map={"": 0},
            quantization_config=quantization_config
        )
    else:
        model = MagmaForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model.to("cuda")

    processor = MagmaProcessor.from_pretrained(base_model, trust_remote_code=True)

    # Load LoRA adapters
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading LoRA adapters from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print("LoRA adapters loaded!")

    model.eval()
    return model, processor


def run_inference(model, processor, image: Image.Image, prompt: str) -> str:
    """Run inference on a single sample."""
    convs = [
        {"role": "system", "content": "You are agent that can see, talk and act."},
        {"role": "user", "content": f"<image_start><image><image_end>\n{prompt}"},
    ]

    prompt_text = processor.tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(images=[image], texts=prompt_text, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
    inputs = inputs.to("cuda").to(model.dtype if hasattr(model, 'dtype') else torch.float16)

    generation_args = {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
        "use_cache": True,
        "num_beams": 1,
    }

    with torch.inference_mode():
        generate_ids = model.generate(**inputs, **generation_args)

    generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
    response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

    return response


def parse_action(response: str) -> dict:
    """Parse model response to extract action dict."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return {"raw_response": response, "parse_error": True}


def compute_metrics(results: list) -> dict:
    """Compute evaluation metrics."""
    total = len(results)
    action_correct = 0
    element_correct = 0
    value_correct = 0
    overall_correct = 0
    parse_errors = 0

    for r in results:
        gt = r['ground_truth']
        pred = r['prediction']

        if pred.get('parse_error'):
            parse_errors += 1
            continue

        # Action accuracy
        action_match = pred.get('ACTION') == gt.get('ACTION')
        if action_match:
            action_correct += 1

        # Element accuracy (MARK)
        element_match = str(pred.get('MARK')) == str(gt.get('MARK'))
        if element_match:
            element_correct += 1

        # Value accuracy (for TYPE actions)
        value_match = True
        if gt.get('ACTION') == 'TYPE':
            value_match = pred.get('VALUE') == gt.get('VALUE')
            if value_match:
                value_correct += 1

        # Overall accuracy
        if action_match and element_match and (gt.get('ACTION') != 'TYPE' or value_match):
            overall_correct += 1

    valid = total - parse_errors

    metrics = {
        'total_samples': total,
        'valid_samples': valid,
        'parse_errors': parse_errors,
        'action_accuracy': action_correct / valid if valid > 0 else 0,
        'element_accuracy': element_correct / valid if valid > 0 else 0,
        'value_accuracy': value_correct / valid if valid > 0 else 0,
        'overall_accuracy': overall_correct / valid if valid > 0 else 0,
    }

    return metrics


def evaluate_checkpoint(checkpoint_path: str, dataset, processor, base_model: str,
                        max_samples: int = None, use_4bit: bool = True, data_dir: str = None) -> dict:
    """Evaluate a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path}")
    print('='*60)

    # Load model with checkpoint
    model, processor = load_model(checkpoint_path, base_model, use_4bit)

    # Prepare samples
    samples = dataset
    if max_samples:
        samples = dataset[:min(max_samples, len(dataset))]

    # Run inference
    results = []
    for sample in tqdm(samples, desc="Evaluating"):
        # Load image from path
        image_path = os.path.join(data_dir, sample['image'])
        image = Image.open(image_path).convert('RGB')

        user_prompt = sample['conversations'][0]['value'].replace("<image>", "").strip()
        gt_value = sample['conversations'][1]['value']

        # Parse ground truth (handle both string and dict)
        if isinstance(gt_value, str):
            try:
                ground_truth = json.loads(gt_value)
            except json.JSONDecodeError:
                ground_truth = {"raw": gt_value}
        else:
            ground_truth = gt_value

        response = run_inference(model, processor, image, user_prompt)
        prediction = parse_action(response)

        results.append({
            "id": sample['id'],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "raw_response": response
        })

    # Compute metrics
    metrics = compute_metrics(results)
    metrics['checkpoint'] = checkpoint_path

    # Print results
    print(f"\n📊 Results for {os.path.basename(checkpoint_path)}:")
    print(f"   Action Accuracy:  {metrics['action_accuracy']*100:.2f}%")
    print(f"   Element Accuracy: {metrics['element_accuracy']*100:.2f}%")
    print(f"   Value Accuracy:   {metrics['value_accuracy']*100:.2f}%")
    print(f"   Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print(f"   Parse Errors:     {metrics['parse_errors']}/{metrics['total_samples']}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mind2Web checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to single checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing multiple checkpoints")
    parser.add_argument("--eval_all", action="store_true",
                        help="Evaluate all checkpoints in directory")
    parser.add_argument("--base_model", type=str, default="microsoft/Magma-8B",
                        help="Base model name")
    parser.add_argument("--data", type=str, default="datasets/mind2web",
                        help="Path to Mind2Web dataset")
    parser.add_argument("--output_dir", type=str, default="results/mind2web_eval",
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset from JSON
    print(f"Loading dataset from: {args.data}")

    # Determine the JSON file path and data directory
    if args.data.endswith('.json'):
        json_path = args.data
        data_dir = os.path.dirname(args.data)
    else:
        data_dir = args.data
        # Try to find validation or test JSON
        val_json = os.path.join(args.data, "mind2web_val.json")
        test_json = os.path.join(args.data, "mind2web_test.json")
        train_json = os.path.join(args.data, "mind2web_train.json")

        if os.path.exists(val_json):
            json_path = val_json
        elif os.path.exists(test_json):
            json_path = test_json
        elif os.path.exists(train_json):
            print("Warning: Using train.json for evaluation (no val/test found)")
            json_path = train_json
        else:
            raise FileNotFoundError(f"No JSON files found in {args.data}")

    print(f"Loading from: {json_path}")
    with open(json_path, 'r') as f:
        data_list = json.load(f)

    # Store data_dir for image loading
    args.data_dir = data_dir

    dataset = data_list
    print(f"Dataset size: {len(dataset)}")

    # Load processor once
    processor = MagmaProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    all_metrics = []

    if args.eval_all and args.checkpoint_dir:
        # Find all checkpoints
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, "checkpoint-*")))
        print(f"Found {len(checkpoints)} checkpoints")

        for ckpt in checkpoints:
            metrics, results = evaluate_checkpoint(
                ckpt, dataset, processor, args.base_model, args.max_samples, args.use_4bit, args.data_dir
            )
            all_metrics.append(metrics)

            # Save individual results
            ckpt_name = os.path.basename(ckpt)
            with open(os.path.join(args.output_dir, f"{ckpt_name}_results.json"), 'w') as f:
                json.dump({"metrics": metrics, "predictions": results}, f, indent=2)

    elif args.checkpoint:
        metrics, results = evaluate_checkpoint(
            args.checkpoint, dataset, processor, args.base_model, args.max_samples, args.use_4bit, args.data_dir
        )
        all_metrics.append(metrics)

        ckpt_name = os.path.basename(args.checkpoint)
        with open(os.path.join(args.output_dir, f"{ckpt_name}_results.json"), 'w') as f:
            json.dump({"metrics": metrics, "predictions": results}, f, indent=2)

    else:
        print("Please provide --checkpoint or --checkpoint_dir with --eval_all")
        return

    # Rank checkpoints by overall accuracy
    if len(all_metrics) > 1:
        print("\n" + "="*60)
        print("📈 CHECKPOINT RANKING (by Overall Accuracy)")
        print("="*60)

        ranked = sorted(all_metrics, key=lambda x: x['overall_accuracy'], reverse=True)

        for i, m in enumerate(ranked):
            ckpt_name = os.path.basename(m['checkpoint'])
            print(f"{i+1}. {ckpt_name}: {m['overall_accuracy']*100:.2f}%")

        best = ranked[0]
        print(f"\n🏆 BEST CHECKPOINT: {os.path.basename(best['checkpoint'])}")
        print(f"   Overall Accuracy: {best['overall_accuracy']*100:.2f}%")

        # Save ranking
        with open(os.path.join(args.output_dir, "checkpoint_ranking.json"), 'w') as f:
            json.dump(ranked, f, indent=2)


if __name__ == "__main__":
    main()
