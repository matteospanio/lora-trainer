import json
import random
from pathlib import Path

OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_TRAIN = 1000
NUM_VAL = 200
NUM_TEST = 200

def generate_input():
    adjectives = ["fast", "slow", "bright", "dark", "smart", "ancient"]
    nouns = ["sensor", "robot", "server", "dataset", "algorithm"]
    return f"{random.choice(adjectives)} {random.choice(nouns)}"

def generate_response(input_str):
    actions = ["measures", "processes", "analyzes", "stores", "monitors"]
    objects = ["temperature", "traffic", "signals", "data", "events"]
    return f"{input_str} {random.choice(actions)} {random.choice(objects)}"

def save_jsonl(data, filename):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def generate_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        inp = generate_input()
        resp = generate_response(inp)
        dataset.append({"prompt": inp, "response": resp})
    return dataset

train_data = generate_dataset(NUM_TRAIN)
val_data = generate_dataset(NUM_VAL)
test_data = generate_dataset(NUM_TEST)

# Save to JSONL files
save_jsonl(train_data, OUTPUT_DIR / "train.jsonl")
save_jsonl(val_data, OUTPUT_DIR / "val.jsonl")
save_jsonl(test_data, OUTPUT_DIR / "test.jsonl")

print(f"Dataset generated in {OUTPUT_DIR}")
print(f"Train: {NUM_TRAIN}, Validation: {NUM_VAL}, Test: {NUM_TEST}")

