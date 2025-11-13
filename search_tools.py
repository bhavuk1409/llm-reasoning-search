import re
import time
from typing import List, Dict, Optional, Tuple
from exa_py import Exa
import logging

logger = logging.getLogger(__name__)

THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
SEARCH_RE = re.compile(r'<search>(.*?)</search>', re.DOTALL | re.IGNORECASE)
RESULT_RE = re.compile(r'<result>(.*?)</result>', re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
NODE_RE = re.compile(r'([A-Z]):\s*(.+?)(?:\((\w+)\))?$', re.MULTILINE)
EDGE_RE = re.compile(r'([A-Z])\s*->\s*([A-Z])', re.IGNORECASE)

def check_r1(text: str) -> bool:
    has_think = bool(THINK_RE.search(text))
    has_search = bool(SEARCH_RE.search(text))
    has_result = bool(RESULT_RE.search(text))
    has_answer = bool(ANSWER_RE.search(text))
    return has_think and has_search and has_result and has_answer

def check_dag(text: str) -> bool:
    search_match = SEARCH_RE.search(text)
    if not search_match:
        return False
    
    search_content = search_match.group(1)
    
    if 'Nodes:' not in search_content or 'Edges:' not in search_content:
        return False
    
    nodes = NODE_RE.findall(search_content)
    if not nodes or len(nodes) > 8:
        return False
    
    node_ids = {n[0] for n in nodes}
    
    edges = EDGE_RE.findall(search_content)
    for src, dst in edges:
        if src not in node_ids or dst not in node_ids:
            return False
        if src == dst:
            return False
    
    return True

def safe_chat_template(tokenizer, messages, add_generation_prompt=True):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
    except Exception:
        result = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                result += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                result += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                result += f"<|im_start|>assistant\n{content}"
        if add_generation_prompt:
            result += "<|im_start|>assistant\n"
        return result

class SearchTools:
    def __init__(self, exa_api_key: str):
        self.exa = Exa(api_key=exa_api_key)
    
    def search(self, query: str, tool: str = "General", num_results: int = 2) -> List[Dict]:
        try:
            results = self.exa.search_and_contents(
                query,
                num_results=num_results,
                use_autoprompt=True,
                text=True
            )
            
            passages = []
            for i, result in enumerate(results.results):
                passages.append({
                    'title': result.title or f"Result {i+1}",
                    'snippet': (result.text or '')[:500],
                    'url': result.url or ''
                })
            
            return passages
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

class DAGParser:
    @staticmethod
    def parse(search_text: str) -> Tuple[List[Tuple], List[Tuple]]:
        nodes = []
        edges = []
        
        if 'Nodes:' in search_text:
            nodes_section = search_text.split('Nodes:')[1]
            if 'Edges:' in nodes_section:
                nodes_section = nodes_section.split('Edges:')[0]
            
            for match in NODE_RE.finditer(nodes_section):
                node_id, query, tool = match.groups()
                tool = tool or "General"
                nodes.append((node_id, query.strip(), tool))
        
        if 'Edges:' in search_text:
            edges_section = search_text.split('Edges:')[1]
            for match in EDGE_RE.finditer(edges_section):
                src, dst = match.groups()
                edges.append((src.upper(), dst.upper()))
        
        return nodes, edges
    
    @staticmethod
    def topological_sort(nodes: List[Tuple], edges: List[Tuple]) -> List[str]:
        from collections import defaultdict, deque
        
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        node_ids = {n[0] for n in nodes}
        for nid in node_ids:
            in_degree[nid] = 0
        
        for src, dst in edges:
            if src in node_ids and dst in node_ids:
                graph[src].append(dst)
                in_degree[dst] += 1
        
        queue = deque([nid for nid in node_ids if in_degree[nid] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(node_ids):
            return sorted(node_ids)
        
        return result

class GenerationManager:
    def __init__(self, tokenizer, model, search_tools, max_turns=2, system_prompt=""):
        self.tokenizer = tokenizer
        self.model = model
        self.search_tools = search_tools
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.trainer = None
    
    def set_trainer_reference(self, trainer):
        self.trainer = trainer
    
    def generate_with_search(self, prompts: List[str], max_completion_length: int = 1024) -> Tuple[List[str], List[str]]:
        prompts_text = []
        completions_text = []
        
        for prompt in prompts:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "<think>\n"}
            ]
            
            prompt_text = safe_chat_template(self.tokenizer, messages, add_generation_prompt=False)
            
            full_completion = self._generate_until_search(prompt_text, max_completion_length)
            
            search_match = SEARCH_RE.search(full_completion)
            if search_match:
                search_content = search_match.group(1)
                nodes, edges = DAGParser.parse(search_content)
                
                if nodes and check_dag(full_completion):
                    result_text = self._execute_search(nodes, edges)
                else:
                    result_text = "\nNo valid search plan generated.\n"
            else:
                result_text = "\nNo search section found.\n"
            
            result_block = f"<result>{result_text}</result>\n<answer>\n"
            
            prompt_with_result = prompt_text + full_completion + result_block
            
            final_answer = self._generate_answer(prompt_with_result, max_completion_length // 2)
            
            completion = full_completion + result_block + final_answer
            
            prompts_text.append(prompt_text)
            completions_text.append(completion)
        
        return prompts_text, completions_text
    
    def _generate_until_search(self, prompt: str, max_length: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens(self.tokenizer, ["</search>"])])
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        return generated
    
    def _generate_answer(self, prompt: str, max_length: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens(self.tokenizer, ["</answer>"])])
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        return generated
    
    def _execute_search(self, nodes: List[Tuple], edges: List[Tuple]) -> str:
        sorted_nodes = DAGParser.topological_sort(nodes, edges)
        node_dict = {n[0]: (n[1], n[2]) for n in nodes}
        
        results_text = ""
        for node_id in sorted_nodes:
            if node_id not in node_dict:
                continue
            
            query, tool = node_dict[node_id]
            passages = self.search_tools.search(query, tool, num_results=2)
            
            results_text += f"\nNode {node_id}: {query}\n"
            for i, passage in enumerate(passages, 1):
                results_text += f"[{i}] {passage['title']}\n{passage['snippet']}\n"
        
        return results_text

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
    
    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        return any(stop in decoded for stop in self.stop_strings)
    
    def __len__(self):
        return 1
    
    def __iter__(self):
        yield self