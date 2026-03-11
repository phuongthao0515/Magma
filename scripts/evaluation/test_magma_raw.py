"""Quick test: run raw Magma-8B on a single mind2web sample.
Matches the official README/app.py pattern exactly.
"""
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

base_model = "./models/Magma-8B"
dtype = torch.bfloat16

# Load model with 4-bit (same as official but quantized to fit 16GB)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    dtype=dtype,
    device_map={"": 0},
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
model.eval()

# Load a mind2web sample
with open("datasets/mind2web/mind2web_val.json") as f:
    data = json.load(f)

sample = data[0]
image_path = f"datasets/mind2web/{sample['image']}"
image = Image.open(image_path).convert("RGB")

user_prompt = sample['conversations'][0]['value'].replace("<image>", "").strip()
gt = sample['conversations'][1]['value']

print(f"Image: {image_path} ({image.size})")
print(f"GT: {gt}")
print(f"Prompt (first 200 chars): {user_prompt[:200]}...")

# Build prompt exactly like official app.py
prompt = f"<image>\n{user_prompt}"
if model.config.mm_use_image_start_end:
    prompt = prompt.replace('<image>', '<image_start><image><image_end>')

convs = [
    {"role": "system", "content": "You are agent that can see, talk and act."},
    {"role": "user", "content": prompt},
]
prompt_text = processor.tokenizer.apply_chat_template(
    convs, tokenize=False, add_generation_prompt=True
)

print(f"\nFormatted prompt (first 300 chars):\n{prompt_text[:300]}...")

# Process inputs exactly like official app.py
inputs = processor(images=[image], texts=prompt_text, return_tensors="pt")
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
inputs = inputs.to(dtype).to("cuda")

print(f"\ninput_ids shape: {inputs['input_ids'].shape}, dtype: {inputs['input_ids'].dtype}")
print(f"pixel_values shape: {inputs['pixel_values'].shape}, dtype: {inputs['pixel_values'].dtype}")
print(f"image_sizes: {inputs['image_sizes']}")

# Generate exactly like official app.py
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        temperature=0.0,
        do_sample=False,
        num_beams=1,
        max_new_tokens=128,
        use_cache=True
    )

# Decode exactly like official app.py
prompt_decoded = processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0]
response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
response = response.replace(prompt_decoded, '').strip()

print(f"\n{'='*60}")
print(f"RAW RESPONSE: {repr(response[:500])}")
print(f"{'='*60}")
