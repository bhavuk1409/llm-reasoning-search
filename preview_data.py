#!/usr/bin/env python3

from data_loader import load_training_data
import json

print("Loading HotpotQA samples...")
data = load_training_data(
    source="hotpotqa",
    max_samples=5,
    difficulty="hard",
    split="train"
)

print(f"\nLoaded {len(data)} samples\n")
print("="*80)

for i, item in enumerate(data, 1):
    print(f"\nExample {i}:")
    print(f"ID: {item['id']}")
    print(f"Question: {item['question']}")
    print(f"Answer: {item['answer']}")
    print(f"Level: {item['level']}")
    print("-"*80)
