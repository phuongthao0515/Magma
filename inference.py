"""
Inference script for Magma-8B
Supports both full precision (17GB) and 4-bit quantization (7GB)
"""

import argparse
from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig

def main(args):
    # Setup quantization if requested
    quantization_config = None
    if args.use_4bit:
        print("Using 4-bit quantization (~7GB VRAM)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        print("Using full precision (~17GB VRAM)")

    print(f"Loading model from {args.model_path}...")
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map={"": 0} if args.use_4bit else None,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
        quantization_config=quantization_config
    )
    if not args.use_4bit:
        model.to("cuda")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print("Model loaded successfully!")

    # Load image
    print(f"Loading image from {args.image_path}...")
    image = Image.open(args.image_path).convert("RGB")

    # Prepare conversation
    convs = [
        {"role": "system", "content": "You are agent that can see, talk and act."},
        {"role": "user", "content": f"<image_start><image><image_end>\n{args.question}"},
    ]

    # Process inputs
    prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], texts=prompt, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)

    # Move to device
    if args.use_4bit:
        inputs = {k: v.to(device=model.device, dtype=torch.float16 if v.dtype == torch.float32 else v.dtype)
                  for k, v in inputs.items()}
    else:
        inputs = inputs.to("cuda").to(torch.bfloat16)

    # Generation settings
    generation_args = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
        "use_cache": True,
        "num_beams": 1,
    }

    print("Generating response...")
    with torch.inference_mode():
        generate_ids = model.generate(**inputs, **generation_args)

    generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
    response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

    # Print results
    print("\n" + "="*70)
    print("IMAGE:", args.image_path)
    print("QUESTION:", args.question)
    print("="*70)
    print("RESPONSE:")
    print(response)
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Magma inference")
    parser.add_argument("--model_path", type=str, default="microsoft/Magma-8B",
                        help="Path to model (default: microsoft/Magma-8B)")
    parser.add_argument("--image_path", type=str, default="assets/images/magma_logo.jpg",
                        help="Path to input image")
    parser.add_argument("--question", type=str, default="What is the letter on the robot?",
                        help="Question to ask about the image")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization (7GB) instead of full precision (17GB)")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy)")

    args = parser.parse_args()
    main(args)
