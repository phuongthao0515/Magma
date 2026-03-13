"""Debug script to find why loss = 0.0 during Mind2Web training."""
import json, copy
from magma.processing_magma import MagmaProcessor

IGNORE_INDEX = -100

processor = MagmaProcessor.from_pretrained('microsoft/Magma-8B', trust_remote_code=True)
tokenizer = processor.tokenizer

with open('datasets/mind2web/mind2web_train.json') as f:
    data = json.load(f)
sample = data[0]

# === Simulate preprocess() ===
convs = copy.deepcopy(sample['conversations'])
for elem in convs:
    elem['role'] = 'user' if elem['from'] in ['human', 'user'] else 'assistant'
    elem['content'] = elem['value']
convs = [{'role': 'system', 'content': 'You are agent that can see, talk and act.'}] + convs

text = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=False)

# Dummy for sep extraction
dummy_convs = [
    {'role': 'system', 'content': 'You are agent that can see, talk and act.'},
    {'role': 'user', 'content': ''},
    {'role': 'assistant', 'content': ''}
]
dummy_text = tokenizer.apply_chat_template(dummy_convs, tokenize=False, add_generation_prompt=False)

empty_token_length = len(tokenizer('').input_ids)
print(f'empty_token_length: {empty_token_length}')

bos_token = tokenizer.bos_token
eos_token = tokenizer.eos_token

segments = dummy_text.split(eos_token)[:-1]
sep1, sep2 = segments[-2], segments[-1]
if bos_token:
    sep1 = sep1.replace(bos_token, '')
print(f'sep1: {repr(sep1)}')
print(f'sep2: {repr(sep2)}')
print()

# Tokenize the full conversation
input_ids = tokenizer(text, return_tensors='pt', padding='longest', max_length=tokenizer.model_max_length, truncation=True).input_ids
target = input_ids.clone()[0]
total_len = int(target.ne(tokenizer.pad_token_id).sum())
print(f'total_len (non-pad tokens): {total_len}')
print(f'input_ids length: {len(target)}')

# Label masking logic (reproduce preprocess() exactly)
conversation = text
conversation_sys = conversation.split(sep1)[0]
conversation_after_sys = conversation[len(conversation_sys):]
rounds = conversation_after_sys.split(sep1)[1:]

cur_len = len(tokenizer(conversation_sys).input_ids)
print(f'sys tokens (cur_len after sys): {cur_len}')
target[:cur_len] = IGNORE_INDEX

print(f'Number of rounds: {len(rounds)}')

for i, rou in enumerate(rounds):
    if rou == '':
        print(f'Round {i}: EMPTY, breaking')
        break
    parts = rou.split(sep2)
    print(f'Round {i}: {len(parts)} parts')
    if len(parts) != 2:
        print(f'  BREAK: parts != 2, parts={[repr(p[:50]) for p in parts]}')
        break
    parts[0] = sep1 + parts[0] + sep2
    rou_full = sep1 + rou

    round_len = len(tokenizer(rou_full).input_ids) - empty_token_length
    instruction_len = len(tokenizer(parts[0]).input_ids) - empty_token_length
    print(f'  round_len: {round_len}')
    print(f'  instruction_len: {instruction_len}')
    print(f'  assistant response tokens: {round_len - instruction_len}')

    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
    cur_len += round_len

target[cur_len:] = IGNORE_INDEX
print()
print(f'cur_len: {cur_len}')
print(f'total_len: {total_len}')
print(f'MATCH: {cur_len == total_len}')

if cur_len != total_len:
    print()
    print('*** TOKENIZATION MISMATCH! All labels would be set to -100! ***')
    print('*** This is why loss = 0.0 ***')
    print()
    # Debug: check what tokens are around cur_len
    print(f'Tokens around cur_len ({cur_len}):')
    start = max(0, cur_len - 5)
    end = min(len(input_ids[0]), cur_len + 5)
    for j in range(start, end):
        tok_id = input_ids[0][j].item()
        tok = tokenizer.decode([tok_id])
        marker = ' <-- cur_len' if j == cur_len else ''
        print(f'  [{j}] id={tok_id} token={repr(tok)}{marker}')

# Final check
num_valid = (target != IGNORE_INDEX).sum().item()
print()
print(f'Valid label tokens (not -100): {num_valid}')
print(f'Total tokens: {len(target)}')
if num_valid == 0:
    print('ALL LABELS ARE -100! Loss will be 0!')
else:
    # Show what tokens are valid labels
    valid_indices = (target != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    print(f'First 20 valid label tokens:')
    for idx in valid_indices[:20]:
        tok_id = target[idx].item()
        tok = tokenizer.decode([tok_id])
        print(f'  [{idx.item()}] id={tok_id} token={repr(tok)}')
