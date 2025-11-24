import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL_NAME = os.environ.get("MODEL_NAME")
LORA_WEIGHTS_PATH = os.environ.get("ARTIFACT_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")

assert LORA_WEIGHTS_PATH is not None, "Set the ARTIFACT_DIR environment variable."
assert BASE_MODEL_NAME is not None, "Set the MODEL_NAME environment variable."
assert CACHE_DIR is not None, "Set the CACHE_DIR environment variable."

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Per batch inference

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=CACHE_DIR,
)

model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
model.eval()


def generate_batch(prompts, max_new_tokens=256, temperature=0.7, batch_size=4):
    """
    Generate responses for multiple prompts in batch
    """
    all_responses = []
    
    # Processa in batch
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating batches"):
        batch_prompts = prompts[i:i + batch_size]
        
        # Formatta i prompt
        formatted = [f"Prompt: {p}\nResponse:" for p in batch_prompts]
        
        # Tokenizza
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Genera
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decodifica
        for output in outputs:
            full_text = tokenizer.decode(output, skip_special_tokens=True)
            response = full_text.split("Response:")[-1].strip()
            all_responses.append(response)
    
    return all_responses


if __name__ == "__main__":
    prompts = [
        "what does dark server?",
        "bright robot",
    ]
    
    print("Generating responses in batch...")
    responses = generate_batch(prompts, batch_size=2)
    
    # Stampa risultati
    print("\n" + "=" * 70)
    for prompt, response in zip(prompts, responses):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 70)
    
    # Salva su file
    with open("batch_results.txt", "w", encoding="utf-8") as f:
        for prompt, response in zip(prompts, responses):
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {response}\n")
            f.write("-" * 70 + "\n\n")
    
    print("\nResults saved in 'batch_results.txt'")
