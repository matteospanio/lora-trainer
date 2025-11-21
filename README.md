# lora-trainer

This repository provides tools for fine-tuning large language models using Low-Rank Adaptation (LoRA) techniques. It includes scripts for training, inference, and synthetic dataset generation.

## Features
- Fine-tune pre-trained language models with LoRA using PEFT and Transformers libraries.
- Generate synthetic datasets for training and evaluation.
- Batch inference for generating responses from multiple prompts.

## Installation
1. Clone the repository
2. Install the required packages:
   ```bash
   uv sync
   ```

## Usage

Before running the scripts you need to create a `.env` file with the same structure as `.env.example` and set the appropriate environment variables.

### Training
To train a model using LoRA, run the `train.py` script with the desired parameters

### Inference
To generate responses from a fine-tuned model, use the `infer.py` script. You can provide multiple prompts for batch generation.

### Synthetic Dataset Generation
Use the `fake_dataset.py` script to create synthetic datasets for training and evaluation.

## Acknowledgements
This project utilizes the PEFT and Transformers libraries. Special thanks to the open-source community for their contributions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
