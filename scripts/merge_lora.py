"""
Merge mind2web LoRA adapter into Magma-8B base model (CPU-only, no GPU needed).

Usage:
    python scripts/merge_lora.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

BASE_MODEL = "microsoft/Magma-8B"
LORA_PATH = "/home/thaole/thao_le/Magma/checkpoints/finetune-mind2web-qlora-768-img-size/checkpoint-1655"
NON_LORA_PATH = "/home/thaole/thao_le/Magma/checkpoints/finetune-mind2web-qlora-768-img-size/non_lora_trainables.bin"
OUTPUT_DIR = "/home/thaole/thao_le/Magma/checkpoints/magma-8b-mind2web-merged"


def main():
    print(f"Loading base model: {BASE_MODEL} (bf16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA from: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH, device_map="cpu")

    print("Merging LoRA into base weights...")
    model = model.merge_and_unload()

    if os.path.exists(NON_LORA_PATH):
        print(f"Loading non-LoRA trainables from: {NON_LORA_PATH}")
        non_lora_state = torch.load(NON_LORA_PATH, map_location="cpu")
        if non_lora_state:
            cleaned = {}
            for k, v in non_lora_state.items():
                clean_key = k.replace("base_model.model.", "").replace("model.", "", 1)
                cleaned[clean_key] = v
            info = model.load_state_dict(cleaned, strict=False)
            print(f"  Loaded {len(cleaned)} weights")

    print(f"Saving merged model to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

    print("Saving tokenizer and image processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.tokenizer.save_pretrained(OUTPUT_DIR)
    processor.image_processor.save_pretrained(OUTPUT_DIR)

    print("Done!")


if __name__ == "__main__":
    main()
