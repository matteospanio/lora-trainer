import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import wandb

wandb.init(
    project="mistral-lora-finetuning",  # Nome del progetto
    name="mistral-7b-lora-run",  # Nome del run
    config={
        "model_name": "mistralai/Mistral-7B-v0.1",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 1000,
    }
)

MODEL_NAME = os.environ.get("MODEL_NAME")
LORA_WEIGHTS_PATH = os.environ.get("ARTIFACT_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")

assert LORA_WEIGHTS_PATH is not None, "Set the ARTIFACT_DIR environment variable."
assert MODEL_NAME is not None, "Set the MODEL_NAME environment variable."
assert CACHE_DIR is not None, "Set the CACHE_DIR environment variable."

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=CACHE_DIR,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_dir="./data", data_files={
    "train": "train.jsonl",
    "validation": "val.jsonl",
    "test": "test.jsonl"
})

def tokenize(batch):
    texts = [
        f"Prompt: {p}\nResponse: {r}"
        for p, r in zip(batch["prompt"], batch["response"])
    ]
    tokens = tokenizer(
        texts,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

    # Mask prompt tokens
    labels = []
    for i, p in enumerate(batch["prompt"]):
        prompt_text = f"Prompt: {p}\nResponse: "
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        
        label = tokens["input_ids"][i].copy()
        label[:prompt_len] = [-100] * prompt_len
        labels.append(label)
    
    tokens["labels"] = labels
    return tokens


tokenized = dataset.map(
    tokenize, 
    batched=True, 
    remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

output_dir = LORA_WEIGHTS_PATH

args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    max_steps=1000,
    learning_rate=2e-4,
    bf16=True,  
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    eval_steps=100,
    output_dir=output_dir,
    optim="adamw_torch",
    report_to="wandb",
    run_name="mistral-7b-lora-run",
    logging_first_step=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

trainer.train()

# Save adapters and tokenizer
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)  # AGGIUNTO
print(f"LoRA adapters and tokenizer saved to {output_dir}")

print("Evaluating on test set...")
test_results = trainer.evaluate(eval_dataset=tokenized["test"])
print(f"Test loss: {test_results['eval_loss']:.4f}")
print(f"Test perplexity: {torch.exp(torch.tensor(test_results['eval_loss'])):.4f}")
