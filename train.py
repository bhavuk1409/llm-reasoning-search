import os
import json
import random
import torch
import logging
import argparse
import openai
import wandb
from typing import Dict, List
from functools import lru_cache

from search_tools import (
    SearchTools, GenerationManager, check_r1, check_dag,
    ANSWER_RE, safe_chat_template
)
from data_loader import load_training_data

from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a search planning agent. For any user question, you must output exactly 4 sections in order:

<think> ... </think>
<search> ... </search>
<r> ... </r>
<answer> ... </answer>

Rules:
1. <think>: Analyze the question, identify key concepts, determine which search sources to use, and plan the search strategy.

2. <search>: Must contain "Nodes:" and "Edges:" sections:
   - Nodes: One per line as "Label: query description (Tool)"
   - Edges: Dependencies as "A -> B; C -> B"
   
3. Constraints:
   - Max 8 nodes
   - Max 10 words per query
   - Tools: General, News, or Academy
   - No self-loops or cycles
   
4. <r>: This section is auto-filled by the system. DO NOT write this in your first response.

5. <answer>: Provide the final answer based on search results.

Do not use any other tags or markdown formatting."""

def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    count = torch.sum(~torch.isnan(tensor))
    variance *= count / (count - 1)
    return torch.sqrt(variance)

class ExternalRewardModel:
    def __init__(self, api_key, model="gpt-4o-mini", score_mode="continuous"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.score_mode = score_mode
    
    _CONTINUOUS_PROMPT = """You are an AI grading assistant. Evaluate the answer quality based on the question and reference answer.
Return a score between 0 and 1 (closer to 1 means more correct).
Output only a number with two decimal places, no other text.

Grading criteria:
- 1.00: Answer aligns with reference in facts, data, conclusions, and core points
- 0.75: Contains most correct info but has minor omissions
- 0.50: Half correct, half wrong or missing
- 0.25: Only small amount of correct info
- 0.00: Completely wrong or irrelevant"""
    
    def _judge(self, question: str, answer: str, reference: str) -> float:
        answer_match = ANSWER_RE.search(answer)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            import re
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.S | re.I)
            answer = re.sub(r'<search>.*?</search>', '', answer, flags=re.S | re.I)
            answer = re.sub(r'<r>.*?</r>', '', answer, flags=re.S | re.I)
            answer = answer.strip()
        
        user_msg = f"Question: {question}\n\nReference Answer: {reference}\n\nAnswer to Grade: {answer}\n\nProvide score:"
        
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=4,
                messages=[
                    {"role": "system", "content": self._CONTINUOUS_PROMPT},
                    {"role": "user", "content": user_msg}
                ]
            )
            
            response_text = rsp.choices[0].message.content.strip()
            
            try:
                import re
                score_text = re.sub(r'[^\d.]', '', response_text)
                score = float(score_text)
                score = max(0.0, min(score, 1.0))
            except (ValueError, TypeError):
                logger.warning(f"Cannot parse '{response_text}' as float, defaulting to 0")
                score = 0.0
            
            return score
        except Exception as e:
            logger.error(f"API error: {e}")
            return 0.0
    
    @lru_cache(maxsize=100000)
    def _judge_cached(self, question: str, answer: str, reference: str) -> float:
        return self._judge(question, answer, reference)
    
    def semantic_score(self, question: str, answer: str, reference: str) -> float:
        try:
            score = self._judge_cached(question, answer, reference)
            return max(0.0, min(score, 1.0))
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return 0.0

class SearchAwareGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, generation_manager=None, **kwargs):
        if generation_manager is None:
            raise ValueError("generation_manager is required")
        
        super().__init__(*args, **kwargs)
        self.generation_manager = generation_manager
        self.generation_manager.model = self.model
        self.use_wandb = getattr(self.args, "use_wandb", False)
    
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        sample_ids = [x.get("sample_id", None) for x in inputs]
        
        if hasattr(self.generation_manager, 'set_trainer_reference'):
            self.generation_manager.set_trainer_reference(self)
        
        prompts_text, completions_text = self.generation_manager.generate_with_search(
            prompts,
            max_completion_length=self.max_completion_length,
        )
        
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        )
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        
        if self.max_prompt_length is not None:
            prompt_inputs = {k: v[:, -self.max_prompt_length:] for k, v in prompt_inputs.items()}
        
        completion_inputs = self.processing_class(
            text=completions_text,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False
        )
        completion_inputs = {k: v.to(device) for k, v in completion_inputs.items()}
        
        import re
        RESULT_RE = re.compile(r'<r>(.*?)</r>', re.DOTALL | re.IGNORECASE)
        for b, comp_txt in enumerate(completions_text):
            result_match = RESULT_RE.search(comp_txt)
            if result_match:
                char_start, char_end = result_match.span()
                
                result_prefix_tokens = len(self.processing_class(
                    comp_txt[:char_start],
                    add_special_tokens=False
                )["input_ids"])
                
                result_tokens = len(self.processing_class(
                    comp_txt[char_start:char_end],
                    add_special_tokens=False
                )["input_ids"])
                
                result_start_pos = result_prefix_tokens
                result_end_pos = min(result_start_pos + result_tokens, completion_inputs["attention_mask"].size(1))
                
                if result_start_pos < completion_inputs["attention_mask"].size(1):
                    completion_inputs["attention_mask"][b, result_start_pos:result_end_pos] = 0
        
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            reward_kwargs["sample_id"] = sample_ids
            
            output_reward_func = reward_func(prompts=prompts_text, completions=completions_text, **reward_kwargs)
            output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        from accelerate.utils import gather
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            attention_mask = torch.cat([prompt_inputs["attention_mask"], completion_inputs["attention_mask"]], dim=1)
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_inputs["attention_mask"].sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        
        if self.num_iterations > 1:
            input_ids = torch.cat([prompt_inputs["input_ids"], completion_inputs["input_ids"]], dim=1)
            attention_mask = torch.cat([prompt_inputs["attention_mask"], completion_inputs["attention_mask"]], dim=1)
            logits_to_keep = completion_inputs["input_ids"].size(1)
            
            with torch.no_grad():
                old_per_token_logps = self._get_per_token_logps(
                    self.model, input_ids, attention_mask, logits_to_keep, self.args.per_device_train_batch_size
                )
        else:
            old_per_token_logps = None
        
        return {
            "prompt_ids": prompt_inputs["input_ids"],
            "prompt_mask": prompt_inputs["attention_mask"],
            "completion_ids": completion_inputs["input_ids"],
            "completion_mask": completion_inputs["attention_mask"],
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": len(prompts),
        }

def train(args):
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    if args.use_wandb:
        wandb.login(key=args.wandb_api_key)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"r-search-{args.model_name.split('/')[-1]}",
            config=vars(args)
        )
        logger.info("WandB initialized")
    
    logger.info("Loading dataset...")
    dataset = load_training_data(
        source=args.data_source,
        file_path=args.data_path if args.data_source == "custom" else None,
        max_samples=args.max_samples,
        difficulty=args.difficulty,
        split=args.data_split,
        shuffle=True
    )
    
    dataset_id_map = {}
    for i, item in enumerate(dataset):
        if "id" in item:
            dataset_id_map[item["id"]] = item
        else:
            dataset_id_map[str(i)] = item
    
    logger.info("Initializing search tools...")
    search_tools = SearchTools(exa_api_key=args.exa_api_key)
    
    logger.info("Initializing reward model...")
    judge = ExternalRewardModel(
        api_key=args.openai_api_key,
        model=args.reward_model,
        score_mode=args.score_mode,
    )
    
    logger.info(f"Loading base model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Configuring LoRA...")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    train_model = get_peft_model(base_model, lora_cfg)
    train_model.config.use_cache = False
    
    logger.info("Initializing generation manager...")
    generation_manager = GenerationManager(
        tokenizer=tokenizer,
        model=train_model,
        search_tools=search_tools,
        max_turns=args.max_turns,
        system_prompt=SYSTEM_PROMPT,
    )
    
    train_dataset = Dataset.from_dict({
        "prompt": [d["question"] for d in dataset],
        "reference": [d["answer"] for d in dataset],
        "sample_id": [str(i) for i in range(len(dataset))],
    })
    
    def reward_fn(prompts, completions, sample_id=None, **kw):
        nonlocal trainer
        
        user_questions = []
        references = []
        
        if sample_id is not None:
            for sid in sample_id:
                if sid in dataset_id_map:
                    item = dataset_id_map[sid]
                    user_questions.append(item["question"])
                    references.append(item["answer"])
                else:
                    user_questions.append("")
                    references.append("")
        else:
            for q_idx, _ in enumerate(prompts):
                if q_idx < len(dataset):
                    user_questions.append(dataset[q_idx]["question"])
                    references.append(dataset[q_idx]["answer"])
                else:
                    user_questions.append("")
                    references.append("")
        
        scores = []
        SEMANTIC_W, DAG_W, R1_W = 0.5, 0.25, 0.25
        MAX_SCORE = SEMANTIC_W + DAG_W + R1_W
        
        CURRICULUM_STEPS = args.curriculum_steps
        if trainer and hasattr(trainer, 'state') and trainer.state.global_step < CURRICULUM_STEPS:
            SEMANTIC_W, DAG_W, R1_W = 0.2, 0.4, 0.4
            MAX_SCORE = SEMANTIC_W + DAG_W + R1_W
        
        for i, (p, comp, ref) in enumerate(zip(prompts, completions, references)):
            user_q = user_questions[i] if i < len(user_questions) else ""
            answer_match = ANSWER_RE.search(comp)
            answer_text = answer_match.group(1).strip() if answer_match else comp
            
            r1_ok_flag = check_r1(comp)
            dag_ok_flag = check_dag(comp)
            
            if not dag_ok_flag:
                semantic = 0.0
                dag_ok = 0.0
                r1_ok = R1_W if r1_ok_flag else 0
                raw_score = r1_ok
                normalized_score = raw_score
                scores.append(normalized_score)
                continue
            
            semantic = judge.semantic_score(user_q, answer_text, ref)
            dag_ok = DAG_W
            r1_ok = R1_W if r1_ok_flag else 0
            raw_score = semantic * SEMANTIC_W + dag_ok + r1_ok
            normalized_score = raw_score
            
            scores.append(normalized_score)
        
        return scores
    
    logger.info("Configuring GRPO trainer...")
    grpo_args = GRPOConfig(
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        bf16=True,
        report_to="wandb" if args.use_wandb else None,
        remove_unused_columns=False,
    )
    
    logger.info("Initializing trainer...")
    trainer = SearchAwareGRPOTrainer(
        model=train_model,
        args=grpo_args,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        generation_manager=generation_manager,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    if trainer.accelerator.is_main_process:
        merged_model = trainer.model.merge_and_unload()
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Saving model to {args.output_dir}")
            merged_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        
        if args.push_to_hub and args.hub_model_id:
            logger.info(f"Pushing to Hub: {args.hub_model_id}")
            try:
                from huggingface_hub import login
                if args.hub_token:
                    login(token=args.hub_token)
                
                merged_model.push_to_hub(
                    args.hub_model_id,
                    private=args.hub_private,
                    token=args.hub_token,
                )
                tokenizer.push_to_hub(
                    args.hub_model_id,
                    private=args.hub_private,
                    token=args.hub_token,
                )
                logger.info(f"Model pushed to https://huggingface.co/{args.hub_model_id}")
            except Exception as e:
                logger.error(f"Hub push error: {e}")
        
        if args.use_wandb:
            wandb.finish()
    
    logger.info("Training complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output_dir", type=str, default="output")
    
    parser.add_argument("--data_source", type=str, default="hotpotqa", 
                        choices=["hotpotqa", "custom"],
                        help="Data source: 'hotpotqa' or 'custom'")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to custom data file (only used if data_source='custom')")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples to load")
    parser.add_argument("--difficulty", type=str, default="hard",
                        choices=["easy", "medium", "hard"],
                        help="HotpotQA difficulty level (only for hotpotqa source)")
    parser.add_argument("--data_split", type=str, default="train",
                        help="Dataset split to use")
    
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--reward_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--exa_api_key", type=str, required=True)
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-6)
    
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])
    
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_turns", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--curriculum_steps", type=int, default=30)
    parser.add_argument("--save_steps", type=int, default=50)
    
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="r-search-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    parser.add_argument("--score_mode", type=str, default="continuous")
    
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_private", action="store_true")
    
    args = parser.parse_args()
    
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    train(args)

if __name__ == "__main__":
    main()