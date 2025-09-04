import random
import requests
import time
import re

# Perplexity API call setup
def call_perplexity_api(prompt, api_key):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar-pro",
        "messages": [{"role":"system","content":"You are a helpful assistant."},
                     {"role":"user","content":prompt}],
        "max_tokens":40000,
        "temperature":0.5
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def parse_feedback_for_improvements(feedback):
    adjustments = []
    if "dropout" in feedback.lower():
        adjustments.append("adjust_dropout")
    if "learning rate" in feedback.lower() or "lr" in feedback.lower():
        adjustments.append("adjust_lr")
    if "layer" in feedback.lower() or "architecture" in feedback.lower():
        adjustments.append("adjust_layer")
    return adjustments if adjustments else ["general_tweak"]

def apply_adjustments(code_lines, adjustments):
    lines = code_lines.copy()
    for adj in adjustments:
        if adj == "adjust_dropout":
            for i, line in enumerate(lines):
                if '.Dropout(' in line or 'nn.Dropout(' in line:
                    new_p = round(random.uniform(0.1, 0.4), 2)
                    parts = line.split('(')
                    if len(parts) > 1:
                        lines[i] = parts[0] + f'({new_p})  # adjusted dropout'
                    break
        elif adj == "adjust_lr":
            # Example: find and scale learning rate
            for i, line in enumerate(lines):
                if 'learning_rate' in line or 'lr =' in line:
                    try:
                        old_lr = float(re.findall(r"\d*\.\d+|\d+", line)[0])
                        new_lr = old_lr * random.uniform(0.7, 1.3)
                        lines[i] = f"learning_rate = {new_lr:.6f}  # adjusted learning rate"
                        break
                    except:
                        continue
        elif adj == "adjust_layer":
            for i, line in enumerate(lines):
                if 'nn.Linear' in line:
                    nums = re.findall(r'\d+', line)
                    if len(nums) >= 2:
                        in_sz = int(nums[0])
                        out_sz = int(nums[1])
                        new_out = int(out_sz * random.uniform(0.8, 1.2))
                        lines[i] = f"nn.Linear({in_sz}, {new_out})  # adjusted layer size"
                        break
        elif adj == "general_tweak":
            # Minor random tweak: delete random line or add comment
            if random.random() < 0.5:
                non_empty = [i for i,l in enumerate(lines) if l.strip()]
                if non_empty:
                    idx = random.choice(non_empty)
                    lines.pop(idx)
            else:
                idx = random.randint(0, len(lines)-1)
                lines[idx] += "  # tweak"
    return lines

def iterative_improve(original_code, api_key, filename_prefix="Model", iterations=3):
    current_code = original_code.split('\n')
    results = []

    for it in range(iterations):
        # Step 1: generate randomized variant or improved code
        if it == 0:
            variant_lines = current_code
        else:
            variant_lines = apply_adjustments(current_code, improvement_adjs)

        # Step 2: convert to string and save
        variant_code_str = "\n".join(variant_lines)
        filename = f"{filename_prefix}_iter_{it+1}.py"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(variant_code_str)

        # Step 3: ask Perplexity AI for feedback
        prompt = (f"Evaluate this Python AI neural network code and make improvements, write the whole code:\n\n{variant_code_str}\n")
        try:
            print(f"Iteration {it+1}: Sending code to Perplexity AI...")
            feedback = call_perplexity_api(prompt, api_key)
            print(f"Feedback received:\n{feedback}\n")
        except Exception as e:
            feedback = f"API error: {e}"
            print(feedback)

        # Step 4: parse feedback to decide improvements
        improvement_adjs = parse_feedback_for_improvements(feedback)

        current_code = variant_lines
        results.append((filename, feedback))

        time.sleep(1)  # polite pause to avoid rate limits

    return results

# Example usage (fill in your API key and your full neural net code as a string):
if __name__ == "__main__":
    API_KEY = ""
    your_code = '''
import re
import torch
import numpy as np
from collections import defaultdict, Counter
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    GPT2LMHeadModel, GPT2Tokenizer,
    pipeline
)
import random
import json
from typing import List, Dict, Tuple, Optional

class TransformerBigramPositionalAI:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize with a pretrained transformer model
        Args:
            model_name: Name of the pretrained model (e.g., 'gpt2', 'distilgpt2', 'gpt2-medium')
        """
        print(f"Loading pretrained model: {model_name}")
        
        # Load pretrained transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline('text-generation', 
                                model=self.model, 
                                tokenizer=self.tokenizer,
                                device=0 if torch.cuda.is_available() else -1)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Bigram position system
        self.word_to_position = {}
        self.position_to_word = {}
        self.bigram_associations = defaultdict(lambda: defaultdict(float))
        self.position_bigrams = defaultdict(lambda: defaultdict(float))
        self.verbatim_associations = defaultdict(list)
        self.transformer_bigram_map = defaultdict(lambda: defaultdict(list))
        self.next_position = 0
        
        # Generation parameters
        self.temperature = 0.8
        self.max_length = 100
        self.top_p = 0.9
        self.top_k = 50
        
        print("Model loaded successfully!")
    
    def tokenize_for_positions(self, text: str) -> List[str]:
        """Tokenize text for position mapping (simpler than transformer tokens)"""
        # Use word-level tokenization for position mapping
        words = re.findall(r'\b\w+\b|[.!?;,]', text.lower())
        return [word for word in words if word.strip()]
    
    def get_or_create_position(self, word: str) -> int:
        """Map word to integer position"""
        if word not in self.word_to_position:
            self.word_to_position[word] = self.next_position
            self.position_to_word[self.next_position] = word
            self.next_position += 1
        return self.word_to_position[word]
    
    def build_associations(self, text: str):
        """Build bigram position associations from text"""
        words = self.tokenize_for_positions(text)
        if len(words) < 2:
            return
        
        # Store verbatim associations
        text_clean = text.strip()
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            self.verbatim_associations[bigram].append({
                'context': text_clean,
                'position': i,
                'full_sequence': words[max(0, i-2):i+4]
            })
        
        # Build position-based bigram associations
        positions = [self.get_or_create_position(word) for word in words]
        
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            pos1, pos2 = positions[i], positions[i + 1]
            
            # Word-based associations
            self.bigram_associations[word1][word2] += 1.0
            
            # Position-based associations
            self.position_bigrams[pos1][pos2] += 1.0
            
            # Map transformer tokens to position bigrams
            transformer_tokens = self.tokenizer.encode(f"{word1} {word2}", add_special_tokens=False)
            if len(transformer_tokens) >= 2:
                token_bigram = tuple(transformer_tokens[:2])
                self.transformer_bigram_map[token_bigram][(pos1, pos2)].append((word1, word2))
    
    def expand_bigram_with_transformer(self, word1: str, word2: str, method: str = 'hybrid') -> Optional[str]:
        """Expand bigram using transformer model guided by position associations"""
        
        if method == 'position_only':
            return self._expand_position_only(word1, word2)
        elif method == 'transformer_only':
            return self._expand_transformer_only(word1, word2)
        else:  # hybrid
            return self._expand_hybrid(word1, word2)
    
    def _expand_position_only(self, word1: str, word2: str) -> Optional[str]:
        """Expand using only position-based associations"""
        pos1 = self.word_to_position.get(word1.lower())
        pos2 = self.word_to_position.get(word2.lower())
        
        if pos1 is not None and pos2 is not None:
            if pos2 in self.position_bigrams[pos1]:
                # Find most likely next position
                next_positions = {}
                for pos in self.position_bigrams:
                    if pos2 in self.position_bigrams[pos]:
                        for next_pos, weight in self.position_bigrams[pos].items():
                            next_positions[next_pos] = next_positions.get(next_pos, 0) + weight
                
                if next_positions:
                    # Sample based on weights
                    positions = list(next_positions.keys())
                    weights = list(next_positions.values())
                    next_pos = random.choices(positions, weights=weights, k=1)[0]
                    return self.position_to_word.get(next_pos)
        
        return None
    
    def _expand_transformer_only(self, word1: str, word2: str) -> Optional[str]:
        """Expand using only transformer model"""
        prompt = f"{word1} {word2}"
        
        try:
            # Generate with transformer
            outputs = self.generator(
                prompt,
                max_length=len(self.tokenizer.encode(prompt)) + 5,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            # Extract the next word after the prompt
            words_after_prompt = generated_text[len(prompt):].strip().split()
            if words_after_prompt:
                return words_after_prompt[0].lower()
                
        except Exception as e:
            print(f"Transformer generation error: {e}")
        
        return None
    
    def _expand_hybrid(self, word1: str, word2: str) -> Optional[str]:
        """Hybrid expansion using both position associations and transformer"""
        
        # Get multiple candidates from transformer
        transformer_candidates = []
        prompt = f"{word1} {word2}"
        
        try:
            outputs = self.generator(
                prompt,
                max_length=len(self.tokenizer.encode(prompt)) + 3,
                temperature=self.temperature * 1.2,  # Slightly higher temperature for diversity
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                num_return_sequences=3,  # Get multiple options
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            for output in outputs:
                generated_text = output['generated_text']
                words_after = generated_text[len(prompt):].strip().split()
                if words_after:
                    transformer_candidates.append(words_after[0].lower())
        
        except Exception as e:
            print(f"Transformer generation error: {e}")
        
        # Get position-based candidates
        position_candidates = []
        bigram = (word1.lower(), word2.lower())
        
        # Check verbatim associations first
        if bigram in self.verbatim_associations:
            for assoc in self.verbatim_associations[bigram]:
                seq = assoc['full_sequence']
                pos = assoc['position']
                if pos + 2 < len(seq):
                    position_candidates.append(seq[pos + 2])
        
        # Check learned bigram associations
        if word2.lower() in self.bigram_associations:
            next_words = list(self.bigram_associations[word2.lower()].keys())
            position_candidates.extend(next_words[:3])  # Top 3 associations
        
        # Combine and rank candidates
        all_candidates = transformer_candidates + position_candidates
        if not all_candidates:
            return None
        
        # Score candidates based on both transformer likelihood and position associations
        candidate_scores = defaultdict(float)
        
        for candidate in all_candidates:
            # Base score from frequency
            candidate_scores[candidate] += all_candidates.count(candidate)
            
            # Bonus for position associations
            if candidate in position_candidates:
                candidate_scores[candidate] += 2.0
            
            # Bonus for transformer suggestions
            if candidate in transformer_candidates:
                candidate_scores[candidate] += 1.5
        
        # Select best candidate
        if candidate_scores:
            candidates = list(candidate_scores.keys())
            scores = list(candidate_scores.values())
            return random.choices(candidates, weights=scores, k=1)[0]
        
        return random.choice(all_candidates) if all_candidates else None
    
    def generate_text(self, seed_text: str = "", max_words: int = 50, method: str = 'hybrid') -> str:
        """Generate text using bigram expansion with transformer guidance"""
        
        if not seed_text.strip():
            # Use transformer to generate a good starting point
            try:
                starter_outputs = self.generator(
                    "",
                    max_length=10,
                    temperature=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                seed_text = starter_outputs[0]['generated_text'].strip()
            except:
                seed_text = "The"
        
        words = self.tokenize_for_positions(seed_text)
        if len(words) < 2:
            words = ["the", "quick"]  # Default start
        
        generated_words = words.copy()
        
        for _ in range(max_words - len(words)):
            if len(generated_words) < 2:
                break
            
            # Use last two words for bigram expansion
            word1 = generated_words[-2]
            word2 = generated_words[-1]
            
            next_word = self.expand_bigram_with_transformer(word1, word2, method)
            
            if next_word and next_word not in ['.', '!', '?']:
                generated_words.append(next_word)
                
                # Occasionally add punctuation for natural flow
                if len(generated_words) % 15 == 0 and random.random() < 0.3:
                    generated_words.append('.')
                    
            elif next_word in ['.', '!', '?']:
                generated_words.append(next_word)
                break
            else:
                # Fallback to transformer-only generation
                fallback = self._expand_transformer_only(word1, word2)
                if fallback:
                    generated_words.append(fallback)
                else:
                    break
        
        # Format output
        result = ' '.join(generated_words)
        result = re.sub(r'\s+([.!?;,])', r'\1', result)  # Fix punctuation spacing
        result = '. '.join(sentence.strip().capitalize() for sentence in result.split('.') if sentence.strip())
        
        if not result.endswith(('.', '!', '?')):
            result += '.'
            
        return result
    
    def train_on_text(self, text: str):
        """Add new text to the association database"""
        self.build_associations(text)
        print(f"Added associations from text: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    def set_generation_params(self, temperature: float = None, max_length: int = None, 
                            top_p: float = None, top_k: int = None):
        """Adjust generation parameters"""
        if temperature is not None:
            self.temperature = temperature
        if max_length is not None:
            self.max_length = max_length
        if top_p is not None:
            self.top_p = top_p
        if top_k is not None:
            self.top_k = top_k
        
        print(f"Updated parameters: temp={self.temperature}, max_len={self.max_length}, top_p={self.top_p}, top_k={self.top_k}")
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_name': self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else 'Unknown',
            'vocab_size': len(self.word_to_position),
            'position_bigrams': len(self.position_bigrams),
            'word_bigrams': len(self.bigram_associations),
            'verbatim_associations': len(self.verbatim_associations),
            'transformer_vocab_size': len(self.tokenizer.vocab),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
        }


def main():
    """Interactive demo"""
    print("=== Transformer-Based Bigram Position AI ===")
    print("Initializing...")
    
    # Initialize with a smaller model for demo (you can change to 'gpt2-medium', 'gpt2-large', etc.)
    ai = TransformerBigramPositionalAI("distilgpt2")  # Faster loading
    
    # Add some initial associations
    sample_texts = [
        "The artificial intelligence system processes natural language with remarkable accuracy.",
        "Machine learning algorithms can identify complex patterns in large datasets effectively.",
        "Deep neural networks have revolutionized computer vision and speech recognition tasks.",
        "Transformers represent a significant breakthrough in natural language processing technology.",
        "Scientists continue to develop more efficient methods for training large language models."
    ]
    
    print("Building initial associations...")
    for text in sample_texts:
        ai.train_on_text(text)
    
    print("\nCommands:")
    print("  generate [seed_text]     - Generate text")
    print("  generate <method> <seed> - Generate with method (hybrid/transformer/position)")
    print("  train <text>            - Add training text")
    print("  params <temp> <top_p>   - Set generation parameters")
    print("  info                    - Show model information")
    print("  quit                    - Exit")
    print()
    
    while True:
        try:
            command = input("AI> ").strip()
            
            if not command:
                continue
                
            if command.lower() == 'quit':
                break
                
            elif command.lower() == 'info':
                info = ai.get_model_info()
                print("Model Information:")
                for key, value in info.items():
                    if key == 'model_parameters':
                        print(f"  {key}: {value:,}")
                    else:
                        print(f"  {key}: {value}")
                        
            elif command.startswith('train '):
                text = command[6:]
                ai.train_on_text(text)
                
            elif command.startswith('params '):
                parts = command[7:].split()
                try:
                    if len(parts) >= 1:
                        temp = float(parts[0])
                        top_p = float(parts[1]) if len(parts) >= 2 else None
                        ai.set_generation_params(temperature=temp, top_p=top_p)
                except ValueError:
                    print("Usage: params <temperature> [top_p]")
                    
            elif command.startswith('generate'):
                parts = command.split(None, 2)
                
                method = 'hybrid'
                seed = ""
                
                if len(parts) == 1:
                    # Just "generate"
                    pass
                elif len(parts) == 2:
                    # "generate <seed>" or "generate <method>"
                    if parts[1] in ['hybrid', 'transformer', 'position']:
                        method = parts[1]
                    else:
                        seed = parts[1]
                elif len(parts) == 3:
                    # "generate <method> <seed>"
                    method = parts[1]
                    seed = parts[2]
                
                print(f"Generating text (method: {method})...")
                result = ai.generate_text(seed, max_words=40, method=method)
                print(f"\n📝 Generated: {result}\n")
                
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
'''
    results = iterative_improve(your_code, API_KEY, "SimpleNet", iterations=3)
    for fname, fb in results:
        print(f"Saved {fname} with feedback:\n{fb}\n{'-'*40}")
