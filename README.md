# R-Search: Reinforcement Learning for Multi-Source Search

Implementation of the R-Search framework from "Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models"

## Features

- Single LLM framework with structured output (think/search/result/answer)
- GRPO-based reinforcement learning
- Exa API integration for web search
- WandB tracking
- Hugging Face Hub integration
- LoRA fine-tuning for efficiency
- **Auto-loading HotpotQA dataset**

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Train with HotpotQA (Recommended)

The easiest way to start training - automatically downloads HotpotQA:

```bash
python train.py \
  --data_source "hotpotqa" \
  --max_samples 500 \
  --difficulty "hard" \
  --openai_api_key "YOUR_KEY" \
  --exa_api_key "YOUR_KEY" \
  --use_wandb \
  --wandb_api_key "YOUR_KEY"
```

### Option 2: Train with Custom Data

Prepare your data in JSON/JSONL format:

```json
[
  {
    "id": "1",
    "question": "Your question here",
    "answer": "Reference answer"
  }
]
```

Then train:

```bash
python train.py \
  --data_source "custom" \
  --data_path "your_data.json" \
  --max_samples 500 \
  --openai_api_key "YOUR_KEY" \
  --exa_api_key "YOUR_KEY"
```

## Preview Data

Before training, preview what data will be loaded:

```bash
python preview_data.py
```

Or create a dataset file:

```bash
python data_loader.py \
  --source hotpotqa \
  --max_samples 1000 \
  --difficulty hard \
  --output training_data.json
```

## Training Arguments

### Data Configuration
- `--data_source`: "hotpotqa" or "custom" (default: hotpotqa)
- `--data_path`: Path to custom data (only for custom source)
- `--max_samples`: Maximum samples to load (default: 1000)
- `--difficulty`: HotpotQA difficulty - "easy", "medium", or "hard" (default: hard)
- `--data_split`: Dataset split - "train" or "validation" (default: train)

### Model & Data
- `--model_name`: Base model (default: DeepSeek-R1-Distill-Qwen-7B)
- `--data_path`: Training data file path (required)
- `--output_dir`: Output directory (default: ./output)

### API Keys
- `--openai_api_key`: OpenAI API key (required)
- `--exa_api_key`: Exa API key (required)
- `--wandb_api_key`: WandB API key (optional)

### Training Hyperparameters
- `--batch_size`: Per-device batch size (default: 1)
- `--max_steps`: Maximum training steps (default: 100)
- `--lr`: Learning rate (default: 5e-6)
- `--num_generations`: GRPO generations per query (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `--curriculum_steps`: Format-focused curriculum steps (default: 30)

### LoRA Configuration
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.05)
- `--lora_target_modules`: Target modules (default: q_proj, k_proj, v_proj, o_proj)

### WandB Tracking
- `--use_wandb`: Enable WandB tracking
- `--wandb_project`: WandB project name (default: r-search-training)
- `--wandb_run_name`: WandB run name (optional)

### Hugging Face Hub
- `--push_to_hub`: Push model to HF Hub
- `--hub_model_id`: HF model ID (e.g., username/model-name)
- `--hub_token`: HF access token
- `--hub_private`: Make repository private

## Output Format

The model generates structured outputs:

```
<think>
[Reasoning about the query and search strategy]
</think>

<search>
Nodes:
A: query text (General)
B: another query (News)
Edges:
A -> B
</search>

<r>
[Auto-filled search results]
</r>

<answer>
[Final synthesized answer]
</answer>
```

## Reward Function

The training uses a composite reward:
- 50% semantic accuracy (via GPT-4o-mini judge)
- 25% DAG validity (structural correctness)
- 25% format compliance (all tags present)

During the first `curriculum_steps`, weights shift to 20%/40%/40% to encourage format learning.

## Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-model-path")
tokenizer = AutoTokenizer.from_pretrained("your-model-path")

prompt = "What are recent developments in quantum computing?"
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

## Hardware Requirements

- Minimum: 1x RTX 4090 (24GB VRAM)
- Recommended: Multi-GPU setup for faster training
- Supports mixed precision (bfloat16) and gradient checkpointing

## Citation

```bibtex
@article{shi2025rsearch,
  title={Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models},
  author={Shi, Wentao and Shen, Yiqing},
  journal={arXiv preprint arXiv:2506.08352},
  year={2025}
}
```

## License

See the original paper and model licenses for usage terms.
