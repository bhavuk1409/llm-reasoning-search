import json
import random
from typing import List, Dict
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def load_hotpotqa(split="train", max_samples=1000, difficulty="hard"):
    logger.info(f"Loading HotpotQA dataset (split={split}, max_samples={max_samples}, difficulty={difficulty})")
    
    try:
        dataset = load_dataset("hotpot_qa", "fullwiki", split=split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load HotpotQA: {e}")
        logger.info("Trying distractor setting instead...")
        dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    
    processed_data = []
    
    for idx, item in enumerate(dataset):
        if len(processed_data) >= max_samples:
            break
        
        question = item.get("question", "")
        answer = item.get("answer", "")
        level = item.get("level", "")
        item_type = item.get("type", "")
        
        if not question or not answer:
            continue
        
        if difficulty and level != difficulty:
            continue
        
        processed_data.append({
            "id": f"hotpotqa_{idx}",
            "question": question,
            "answer": answer,
            "type": "multi_hop",
            "level": level,
            "source": "hotpotqa"
        })
    
    logger.info(f"Loaded {len(processed_data)} samples from HotpotQA")
    
    if len(processed_data) < max_samples:
        logger.warning(f"Only found {len(processed_data)} samples (requested {max_samples})")
    
    return processed_data

def load_custom_json(file_path: str) -> List[Dict]:
    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    
    for i, item in enumerate(data):
        if "id" not in item:
            item["id"] = f"custom_{i}"
        if "type" not in item:
            item["type"] = "fact"
        if "source" not in item:
            item["source"] = "custom"
    
    return data

def load_training_data(
    source="hotpotqa",
    file_path=None,
    max_samples=1000,
    difficulty="hard",
    split="train",
    shuffle=True,
    seed=42
):
    if source == "hotpotqa":
        data = load_hotpotqa(
            split=split,
            max_samples=max_samples,
            difficulty=difficulty
        )
    elif source == "custom" and file_path:
        data = load_custom_json(file_path)
        if len(data) > max_samples:
            data = data[:max_samples]
    else:
        raise ValueError(f"Unknown source: {source}. Use 'hotpotqa' or 'custom'")
    
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    
    logger.info(f"Final dataset size: {len(data)} samples")
    
    return data

def save_dataset(data: List[Dict], output_path: str):
    if output_path.endswith('.jsonl'):
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(data)} samples to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="hotpotqa", choices=["hotpotqa", "custom"])
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard", None])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="training_data.json")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    data = load_training_data(
        source=args.source,
        file_path=args.file_path,
        max_samples=args.max_samples,
        difficulty=args.difficulty,
        split=args.split
    )
    
    save_dataset(data, args.output)
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(data)}")
    print(f"\nFirst example:")
    print(json.dumps(data[0], indent=2, ensure_ascii=False))
