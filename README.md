## Multi-Hop Reasoning Project

## Overview
This project develops a Large Language Model (LLM) capable of multi-hop reasoning using reinforcement fine-tuning.  
The model autonomously plans, searches, and synthesizes information from multiple sources within a single framework.

---

## Objective
To train an LLM that can:
1. Plan how to search (queries and sources)
2. Execute those searches
3. Combine retrieved information into a coherent answer

---

## Model Output Format
Each response follows a structured reasoning format:

<think> Internal reasoning process </think>

<search> Multi-hop search plan (DAG) </search> <result> Retrieved information </result> <answer> Final answer </answer> ```
Training Pipeline
1. Data Collection
Datasets used:

FinSearchBench-24

SearchExpertBench-25

Each entry includes:

A question

A correct answer

(Optional) reference search structure

2. Candidate Generation
A base model (Qwen-2 or Mistral) generates structured responses that include <think>, <search>, <result>, and <answer> tags.

3. Reward Computation
Reward Type	Description
Answer Reward	Correctness of the final answer
Format Reward	Structural accuracy of the response
Search Reward	Logical and efficient DAG generation
Efficiency Reward	Conciseness of reasoning and output

Reward Function:

ini
Copy code
R = w1*R_answer + w2*R_format + w3*R_search + w4*R_efficiency
4. Reinforcement Fine-Tuning
The model is fine-tuned using Generalized Reward Policy Optimization (GRPO), optimizing for high-reward structured reasoning.

5. Evaluation
The model is evaluated based on:

Final answer accuracy

Validity of search DAG

Token efficiency

Latency

Results
Dataset	Baseline	Proposed (R-Search)	Improvement
FinSearchBench-24	64.1%	79.2%	+15.1%
SearchExpertBench-25	58.4%	75.3%	+16.9%

Metric	Multi-Agent	R-Search	Improvement
Tokens Used	8.2k	2.4k	~70% fewer
Latency	13.2s	6.7s	~50% faster

References
Yao, S. et al. ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629 (2022)

Schick, T. et al. Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761 (2023)

Shao, Z. et al. VinePPO: Unlocking RL Potential for LLM Reasoning through Refined Credit Assignment. arXiv:2410.01679 (2024)

Li, Y. et al. Reasoning with Retrieval: Augmenting Large Language Models with Information Search. arXiv:2403.10129 (2024)
