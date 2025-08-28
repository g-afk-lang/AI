import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

import numpy as np
import math
from collections import Counter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
import pickle
import random

# Try to import Hugging Face datasets, but make it optional
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

# Global configuration
KB_LEN = 10000  # Maximum number of words to process (-1 for unlimited)

# ------------------------------------------------------
# Utility
# ------------------------------------------------------
def custom_sigmoid(x):
    """Heavy sigmoid function using -5/x formulation with safety handling."""
    x_safe = torch.where(torch.abs(x) > 1e-8, x, torch.sign(x) * 1e-8)
    return torch.sigmoid(-5.0 / x_safe)

# ------------------------------------------------------
# Math Processor
# ------------------------------------------------------
class MathProcessor(nn.Module):
    """Mathematical processor implementing construction principles."""
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.register_parameter('compass_radius_scale', nn.Parameter(torch.tensor(0.5)))
        self.register_parameter('circle_intersection_threshold', nn.Parameter(torch.tensor(0.7)))
        self.register_parameter('geometric_precision', nn.Parameter(torch.tensor(1e-6)))
        
        self.register_buffer('golden_ratio', torch.tensor((1 + math.sqrt(5)) / 2))
        self.register_buffer('pi_approx', torch.tensor(22.0 / 7.0))
        
    def circle_circle_intersection(self, center1, radius1, center2, radius2):
        d = torch.norm(center2 - center1)
        intersect_condition = torch.logical_and(d <= (radius1 + radius2), d >= torch.abs(radius1 - radius2))
        if not intersect_condition.any():
            return torch.zeros(2, 2, device=self.device), torch.tensor(False, device=self.device)
        
        # Basic intersection calculation (simplified)
        intersections = torch.stack([center1, center2])
        return intersections, torch.tensor(True, device=self.device)
    
    def compass_only_midpoint(self, point1, point2):
        center_dist = torch.norm(point2 - point1)
        radius = center_dist * self.compass_radius_scale
        intersections, valid = self.circle_circle_intersection(point1, radius, point2, radius)
        if valid:
            midpoint = (intersections[0] + intersections[1]) / 2
            return midpoint
        else:
            return (point1 + point2) / 2

# ------------------------------------------------------
# Heavy Duty Cycle Manager
# ------------------------------------------------------
class TrainableMemoryOptimizedHeavyDutyCycleManager(nn.Module):
    def __init__(self, cycle_length=32, duty_ratio=0.8, decay_rate=0.7, device='cpu', max_buffer_size=100):
        super().__init__()
        self.register_parameter('cycle_length', nn.Parameter(torch.tensor(float(cycle_length))))
        self.register_parameter('duty_ratio', nn.Parameter(torch.tensor(duty_ratio)))
        self.register_parameter('decay_rate', nn.Parameter(torch.tensor(decay_rate)))
        self.register_parameter('neural_feedback_gain', nn.Parameter(torch.tensor(0.2)))
        self.register_buffer('cycle_position', torch.tensor(0.0, device=device))
        self.max_buffer_size = max_buffer_size
        self.probability_buffer = []
        self.cycle_history = []
        self.register_buffer('thermal_accumulator', torch.tensor(0.0, device=device))
        self.device = device
        self.running_mean = 0.0
        self.running_var = 0.0
        self.sample_count = 0
        self.register_parameter('active_modulation_scale', nn.Parameter(torch.tensor(0.5)))
        self.register_parameter('inactive_modulation_scale', nn.Parameter(torch.tensor(0.1)))
        self.register_parameter('sigmoid_scale', nn.Parameter(torch.tensor(1.0)))
        
    @property
    def active_threshold(self):
        cycle_length_val = self.cycle_length.item()
        duty_ratio_val = self.duty_ratio.item()
        threshold = cycle_length_val * duty_ratio_val
        return torch.clamp(torch.tensor(threshold, device=self.cycle_length.device), 1.0, cycle_length_val - 1.0)
        
    def _update_running_stats(self, value):
        self.sample_count += 1
        delta = value - self.running_mean
        self.running_mean += delta / self.sample_count
        delta2 = value - self.running_mean
        self.running_var += delta * delta2
        
    def reset_state(self):
        """Reset all internal state for clean training batches."""
        self.cycle_position.data.zero_()
        self.thermal_accumulator.data.zero_()
        self.probability_buffer.clear()
        self.cycle_history.clear()
        self.running_mean = 0.0
        self.running_var = 0.0
        self.sample_count = 0
        
    def _prune_buffers(self):
        if len(self.probability_buffer) > self.max_buffer_size:
            self.probability_buffer = self.probability_buffer[-self.max_buffer_size//2:]
        if len(self.cycle_history) > 10:
            self.cycle_history = self.cycle_history[-5:]
        
    def modulate_probabilities(self, base_probabilities, neural_activity=None):
        self.cycle_position += 1.0
        cycle_reset = (self.cycle_position >= self.cycle_length).float()
        self.cycle_position = self.cycle_position * (1 - cycle_reset)
        if cycle_reset.item() > 0:
            self._prune_buffers()
            
        modulation = self.get_duty_cycle_modulation()
        custom_modulation = custom_sigmoid(modulation * self.sigmoid_scale)
        
        if isinstance(base_probabilities, torch.Tensor):
            modulated = base_probabilities * custom_modulation
            avg_prob = modulated.mean().item()
        else:
            base_probs_tensor = torch.tensor(base_probabilities, device=self.device, dtype=torch.float32)
            modulated = base_probs_tensor * custom_modulation
            avg_prob = modulated.mean().item()
        self._update_running_stats(avg_prob)
        if len(self.probability_buffer) < self.max_buffer_size:
            self.probability_buffer.append(avg_prob)
        return modulated
    
    def get_duty_cycle_modulation(self):
        active_thresh = self.active_threshold
        phase_input = 2.0 * (active_thresh - self.cycle_position)
        is_active = custom_sigmoid(phase_input)
        progress = self.cycle_position / torch.clamp(active_thresh, min=1e-8)
        active_mod = self.active_modulation_scale * (1.0 + torch.sin(progress * torch.pi))
        inactive_progress = (self.cycle_position - active_thresh) / torch.clamp(
            self.cycle_length - active_thresh, min=1e-8)
        inactive_mod = self.inactive_modulation_scale * torch.exp(-3.0 * inactive_progress)
        modulation = is_active * active_mod + (1 - is_active) * inactive_mod
        return modulation

# ------------------------------------------------------
# LIF Neuron
# ------------------------------------------------------
class TrainableMemoryEfficientLIFNeuron(nn.Module):
    def __init__(self, tau_mem=10.0, tau_syn=5.0, v_thresh=1.0, v_reset=0.0):
        super().__init__()
        self.register_parameter('tau_mem', nn.Parameter(torch.tensor(tau_mem)))
        self.register_parameter('tau_syn', nn.Parameter(torch.tensor(tau_syn)))
        self.register_parameter('v_thresh', nn.Parameter(torch.tensor(v_thresh)))
        self.register_parameter('v_reset', nn.Parameter(torch.tensor(v_reset)))
        self.register_parameter('sigmoid_gain', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('membrane_nonlinearity', nn.Parameter(torch.tensor(0.1)))
        
    def compute_decay_factors(self):
        tau_mem_clamped = torch.clamp(self.tau_mem, 1.0, 150.0)
        tau_syn_clamped = torch.clamp(self.tau_syn, 1.0, 250.0)
        beta = torch.exp(-1.0 / tau_mem_clamped)
        alpha = torch.exp(-1.0 / tau_syn_clamped)
        return beta, alpha
        
    def forward(self, x, state=None):
        device = x.device
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)
        batch_size, num_neurons = x.shape
        if state is None:
            v_mem = torch.zeros(batch_size, num_neurons, device=device)
            i_syn = torch.zeros(batch_size, num_neurons, device=device)
        else:
            v_mem, i_syn = state
        beta, alpha = self.compute_decay_factors()
        i_syn = alpha * i_syn + x
        membrane_update = i_syn + custom_sigmoid(v_mem * self.membrane_nonlinearity)
        v_mem = beta * v_mem + membrane_update
        thresh_clamped = torch.clamp(self.v_thresh, 0.1, 5.0)
        if self.training:
            spike_input = (v_mem - thresh_clamped) * self.sigmoid_gain
            spike_prob = custom_sigmoid(spike_input)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(spike_prob) + 1e-8) + 1e-8)
            spikes = torch.sigmoid((torch.log(spike_prob + 1e-8) - torch.log(1 - spike_prob + 1e-8) + gumbel_noise) / 0.1)
        else:
            spike_candidates = custom_sigmoid((v_mem - thresh_clamped) * self.sigmoid_gain)
            spikes = (spike_candidates >= 0.5).float()
        reset_clamped = torch.clamp(self.v_reset, -2.0, 2.0)
        reset_strength = custom_sigmoid(spikes * 5.0)
        v_mem = v_mem * (1 - reset_strength) + reset_clamped * reset_strength
        return spikes, (v_mem, i_syn)

# ------------------------------------------------------
# SNN Model
# ------------------------------------------------------
class TrainableStreamingSNN(nn.Module):
    def __init__(self, num_neurons, device='cpu', chunk_size=32):
        super().__init__()
        self.num_neurons = num_neurons
        self.device = device
        self.chunk_size = chunk_size
        self.input_layer = nn.Linear(num_neurons, num_neurons, bias=True)
        self.hidden_layer = nn.Linear(num_neurons, num_neurons, bias=True)
        self.output_layer = nn.Linear(num_neurons, num_neurons, bias=True)
        self.register_parameter('activation_scale1', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('activation_scale2', nn.Parameter(torch.tensor(1.0)))
        self.lif_neurons = TrainableMemoryEfficientLIFNeuron()
        self.global_adaptation = nn.Parameter(torch.ones(1) * 0.5)
        self.duty_cycle_manager = TrainableMemoryOptimizedHeavyDutyCycleManager(device=device)
        self.neuron_state = None
        
    def forward_chunk(self, x_chunk):
        if x_chunk.dim() == 1:
            x_chunk = x_chunk.unsqueeze(0)
        if x_chunk.shape[-1] != self.num_neurons:
            if x_chunk.shape[-1] > self.num_neurons:
                x_chunk = x_chunk[..., :self.num_neurons]
            else:
                padding_size = self.num_neurons - x_chunk.shape[-1]
                padding = torch.zeros(*x_chunk.shape[:-1], padding_size, device=x_chunk.device)
                x_chunk = torch.cat([x_chunk, padding], dim=-1)
        x_processed = custom_sigmoid(self.input_layer(x_chunk) * self.activation_scale1)
        intersect_condition = torch.logical_and(x_chunk <= (x_processed + x_chunk), x_chunk >= torch.abs(x_processed - x_chunk))

        x_hidden = custom_sigmoid(self.hidden_layer(x_processed) * self.activation_scale2)
        prob_weights = custom_sigmoid(x_hidden)
        modulated_weights = self.duty_cycle_manager.modulate_probabilities(prob_weights, neural_activity=intersect_condition)
        x_modulated = x_hidden + modulated_weights.unsqueeze(0)
        spikes, self.neuron_state = self.lif_neurons(x_modulated, self.neuron_state)
        output = custom_sigmoid(self.output_layer(spikes))
        cycle_mod = self.duty_cycle_manager.get_duty_cycle_modulation()
        adapted_output = output * self.global_adaptation * (1 + cycle_mod)
        return adapted_output.squeeze(0)
    
    def forward(self, x_sequence):
        outputs = []
        self.reset_neurons()
        for x in x_sequence:
            out = self.forward_chunk(x)
            outputs.append(out)
        return torch.stack(outputs) if outputs else torch.empty(0, self.num_neurons)
    
    def reset_neurons(self):
        """Reset neuron states and clear computational graph references."""
        if hasattr(self, 'neuron_state') and self.neuron_state is not None:
            # Detach from computational graph
            v_mem, i_syn = self.neuron_state
            self.neuron_state = (v_mem.detach(), i_syn.detach())
        else:
            self.neuron_state = None
        
        # Also reset duty cycle manager state
        if hasattr(self, 'duty_cycle_manager'):
            self.duty_cycle_manager.reset_state()

# ------------------------------------------------------
# Enhanced Text Processor (with n-gram support) - FIXED
# ------------------------------------------------------
class EnhancedTextProcessor(nn.Module):
    def __init__(self, num_neurons=256, device='cpu', vocab_limit=5000, max_features=1000):
        super().__init__()
        self.num_neurons = num_neurons
        self.device = device
        self.vocab_limit = vocab_limit
        self.word_to_idx = {}
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.ngram_cache = {}
        self.math_processor = MathProcessor(device=device)
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z0-9]+\b'
        )
        self.tfidf_scaler = StandardScaler()
        self.is_vectorizer_fitted = False
        self.tfidf_projection = nn.Sequential(
            nn.Linear(max_features, num_neurons // 4),
            nn.Dropout(0.1),
            nn.Linear(num_neurons // 4, num_neurons // 4)
        )
        self.word_embeddings = nn.Embedding(vocab_limit + 1, num_neurons // 4)
        self.position_embeddings = nn.Embedding(1000, num_neurons // 4)
        self.geometric_embeddings = nn.Embedding(100, num_neurons // 4)
        self.compass_feature_processor = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons)
        )
        self.register_parameter('geometric_sigmoid_scale', nn.Parameter(torch.tensor(1.2)))
        self.register_parameter('tfidf_sigmoid_scale', nn.Parameter(torch.tensor(1.0)))
        self.geometric_terms = {
            'compass': 0, 'circle': 1, 'intersection': 2, 'construction': 3,
            'midpoint': 4, 'perpendicular': 5, 'radius': 6, 'center': 7,
            'arc': 8, 'point': 9, 'line': 10, 'geometry': 11,
            'mohr': 12, 'theorem': 13, 'euclidean': 14,
            'straightedge': 15, 'triangle': 16, 'square': 17, 'polygon': 18
        }
        self.transition_cache = {}
        self.cache_limit = 1000
        
    def fit_vectorizer(self, documents):
        print("Fitting TF-IDF vectorizer...")
        processed_docs = []
        for doc in documents:
            if isinstance(doc, list):
                doc = ' '.join(doc)
            processed_docs.append(doc)
        if not processed_docs:
            print("No documents available for vectorizer fitting")
            return
        self.vectorizer.fit(processed_docs)
        tfidf_matrix = self.vectorizer.transform(processed_docs)
        self.tfidf_scaler.fit(tfidf_matrix.toarray())
        self.is_vectorizer_fitted = True
        print(f"Vectorizer fitted with {len(self.vectorizer.get_feature_names_out())} features")
        
    def text_to_tfidf_features(self, text):
        if not self.is_vectorizer_fitted:
            return torch.ones(1, self.tfidf_projection[0].in_features, device=self.device)
        if isinstance(text, list):
            text = ' '.join(text)
        tfidf_matrix = self.vectorizer.transform([text])
        tfidf_features = self.tfidf_scaler.transform(tfidf_matrix.toarray())
        return torch.tensor(tfidf_features, dtype=torch.float32, device=self.device)
    
    def encode_geometric_terms(self, words):
        geometric_indices = []
        for word in words:
            if word.lower() in self.geometric_terms:
                geometric_indices.append(self.geometric_terms[word.lower()])
            else:
                geometric_indices.append(0)
        if geometric_indices:
            geo_indices_tensor = torch.tensor(geometric_indices, device=self.device)
            geometric_features = self.geometric_embeddings(geo_indices_tensor)
            return geometric_features.mean(dim=0, keepdim=True)
        else:
            return torch.zeros(1, self.num_neurons // 4, device=self.device)
    
    def apply_compass_construction_to_features(self, features):
        batch_size, feature_dim = features.shape
        geometric_transform = torch.sin(features + math.pi / 4) + torch.cos(features + math.pi / 6)
        construction_effect = features * 0.9 + geometric_transform * 0.1
        return construction_effect
    
    def words_to_neural_features(self, words, max_words=50):
        if len(words) > max_words:
            words = words[-max_words:]
        device = self.device
        
        tfidf_features = self.text_to_tfidf_features(words)
        expected_size = self.tfidf_projection[0].in_features
        if tfidf_features.shape[1] != expected_size:
            if tfidf_features.shape[1] < expected_size:
                padding = torch.zeros(tfidf_features.shape[0], expected_size - tfidf_features.shape[1], device=device)
                tfidf_features = torch.cat([tfidf_features, padding], dim=1)
            else:
                tfidf_features = tfidf_features[:, :expected_size]
        tfidf_processed = custom_sigmoid(self.tfidf_projection(tfidf_features) * self.tfidf_sigmoid_scale)
        
        word_indices = []
        for word in words:
            idx = self.word_to_idx.get(word, 0)
            word_indices.append(min(idx, self.vocab_limit))
           
        if not word_indices:
            word_features = torch.zeros(1, self.num_neurons // 4, device=device)
        else:
            word_indices = torch.tensor(word_indices, device=device)
            word_embs = self.word_embeddings(word_indices)
            word_features = word_embs.mean(dim=0, keepdim=True)
        
        position_indices = torch.arange(min(len(words), 999), device=device)
        pos_embs = self.position_embeddings(position_indices)
        pos_features = pos_embs.mean(dim=0, keepdim=True)
      
        geo_features = self.encode_geometric_terms(words)
        combined_features = torch.cat([tfidf_processed, word_features, pos_features, geo_features], dim=1)
        compass_features = custom_sigmoid(self.compass_feature_processor(combined_features) * self.geometric_sigmoid_scale)
        final_features = self.apply_compass_construction_to_features(compass_features)
        return final_features
    
    def load_and_process_text_streaming(self, file_path="test.txt", chunk_size=1000, dataset_name=None, split=None):
        word_count = 0
        documents = []
        current_doc = []
        vocab = list(self.geometric_terms.keys())
        for word in vocab:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        words_processed = []
        
        # Add Hugging Face dataset support
        if dataset_name is not None:
            if not HF_DATASETS_AVAILABLE:
                print("Hugging Face datasets library not available. Install with: pip install datasets")
                print("Falling back to local file loading...")
                return self.load_and_process_text_streaming(file_path=file_path, chunk_size=chunk_size)
            
            try:
                print(f"Loading Hugging Face dataset: {dataset_name}, split: {split}")
                ds = load_dataset(dataset_name, split=split or 'train')
                text_field = 'text' if 'text' in ds.column_names else ds.column_names[0]
                
                for text in ds[text_field]:
                    if KB_LEN > 0 and word_count >= KB_LEN:
                        break
                    words = str(text).lower().split()
                    word_history = []
                    for word in words:
                        if len(self.word_to_idx) < self.vocab_limit:
                            if word not in self.word_to_idx:
                                self.word_to_idx[word] = len(self.word_to_idx)
                        
                        word_history.append(word)
                        if len(word_history) >= 2:
                            self.bigram_counts[(word_history[-2], word_history[-1])] += 1
                        if len(word_history) >= 3:
                            self.trigram_counts[(word_history[-3], word_history[-2], word_history[-1])] += 1
                        if len(word_history) > 1000:
                            word_history = word_history[-500:]
                        
                        words_processed.append(word)
                        current_doc.append(word)
                        word_count += 1
                        if len(current_doc) >= 100:
                            documents.append(' '.join(current_doc))
                            current_doc = []
                        if KB_LEN > 0 and word_count >= KB_LEN:
                            break
                            
                if current_doc:
                    documents.append(' '.join(current_doc))
                    
            except Exception as e:
                print(f"Dataset loading failed: {e}. Falling back to local file.")
                return self.load_and_process_text_streaming(file_path=file_path, chunk_size=chunk_size)
        else:
            # Original local file loading logic
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    word_history = []
                    while KB_LEN < 0 or word_count < KB_LEN:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        words = chunk.lower().split()
                        for word in words:
                            if len(self.word_to_idx) < self.vocab_limit:
                                if word not in self.word_to_idx:
                                    self.word_to_idx[word] = len(self.word_to_idx)
                            word_history.append(word)
                            if len(word_history) >= 2:
                                self.bigram_counts[(word_history[-2], word_history[-1])] += 1
                            if len(word_history) >= 3:
                                self.trigram_counts[(word_history[-3], word_history[-2], word_history[-1])] += 1
                            if len(word_history) > 1000:
                                word_history = word_history[-500:]
                            
                            words_processed.append(word)
                            current_doc.append(word)
                            word_count += 1
                            if len(current_doc) >= 100:
                                documents.append(' '.join(current_doc))
                                current_doc = []
                            if KB_LEN > 0 and word_count >= KB_LEN:
                                break
                    if current_doc:
                        documents.append(' '.join(current_doc))
            except FileNotFoundError:
                print(f"File {file_path} not found. Using sample data...")
                sample_words = vocab * 50
                documents = [' '.join(sample_words[i:i+50]) for i in range(0, len(sample_words), 50)]
                for i, word in enumerate(sample_words):
                    if word not in self.word_to_idx:
                        self.word_to_idx[word] = len(self.word_to_idx)
                    if i > 0:
                        self.bigram_counts[(sample_words[i-1], word)] += 1
                words_processed = sample_words

        if documents and not self.is_vectorizer_fitted:
            self.fit_vectorizer(documents)
        print(f"Processed {word_count} words with vocab size {len(self.word_to_idx)}")
        print(f"Created {len(documents)} documents")
        return words_processed[-1000:] if words_processed else []
    
    def get_transition_probs(self, word):
        if word in self.transition_cache:
            return self.transition_cache[word]
        transitions = []
        for (w1, w2), count in self.bigram_counts.items():
            if w1 == word:
                weight_multiplier = 2.0 if w2 in self.geometric_terms else 1.0
                transitions.append((w2, count * weight_multiplier))
        if len(self.transition_cache) >= self.cache_limit:
            keys_to_remove = list(self.transition_cache.keys())[:self.cache_limit//2]
            for k in keys_to_remove:
                del self.transition_cache[k]
        self.transition_cache[word] = transitions
        return transitions
        
    def get_ngram_transitions(self, context_words, n=3):
        if len(context_words) < n - 1:
            return []
        context_key = tuple(context_words[-(n-1):])
        if context_key in self.ngram_cache:
            return self.ngram_cache[context_key]
        transitions = []
        if n == 3:
            for (w1, w2, w3), count in self.trigram_counts.items():
                if (w1, w2) == context_key:
                    weight_multiplier = 2.0 if w3 in self.geometric_terms else 1.0
                    transitions.append((w3, count * weight_multiplier))
        self.ngram_cache[context_key] = transitions
        return transitions

# ------------------------------------------------------
# Text Generator (multi-word seed)
# ------------------------------------------------------
class TrainableStreamingTextGenerator(nn.Module):
    def __init__(self, text_processor, hidden_dim=128, max_transitions_per_word=50):
        super().__init__()
        self.text_processor = text_processor
        self.max_transitions = max_transitions_per_word
        self.fallback_words = ["the", "and", "to", "of", "a", "in", "is", "it", "you", "that"]
        self.selection_network = nn.Sequential(
            nn.Linear(text_processor.num_neurons, hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.register_parameter('selection_sigmoid_scale', nn.Parameter(torch.tensor(1.0)))
        self.context_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, spk_rec):
        if spk_rec.numel() == 0:
            # Return appropriate tensor for empty input
            device = next(self.parameters()).device
            return torch.zeros(1, device=device)
        
        # Ensure proper dimensions for the selection network
        if spk_rec.dim() == 1:
            spk_rec = spk_rec.unsqueeze(0)  # Add batch dimension
        elif spk_rec.dim() > 2:
            # Flatten extra dimensions but keep batch
            batch_size = spk_rec.shape[0]
            spk_rec = spk_rec.view(batch_size, -1)
        
        # Check if input size matches expected input size
        expected_input_size = self.selection_network[0].in_features
        actual_input_size = spk_rec.shape[-1]
        
        if actual_input_size != expected_input_size:
            if actual_input_size < expected_input_size:
                # Pad with zeros
                padding_size = expected_input_size - actual_input_size
                padding = torch.zeros(spk_rec.shape[0], padding_size, device=spk_rec.device)
                spk_rec = torch.cat([spk_rec, padding], dim=-1)
            else:
                # Truncate
                spk_rec = spk_rec[:, :expected_input_size]
        
        linear_output = self.selection_network(spk_rec)
        selection_probs = custom_sigmoid(linear_output.squeeze(-1) * self.selection_sigmoid_scale)
        return selection_probs
    
    def get_multi_word_transitions(self, seed_words):
        if not seed_words:
            return []
        trigram_transitions = self.text_processor.get_ngram_transitions(seed_words, n=3)
        if trigram_transitions:
            return trigram_transitions
        bigram_transitions = self.text_processor.get_transition_probs(seed_words[-1])
        return bigram_transitions
    
    def generate_text_trainable(self, spk_rec, seed_words=None, length=50):
        if spk_rec.numel() == 0:
            return "No neural data available for generation."
        with torch.no_grad():
            selection_probs = self.forward(spk_rec)
        
        if seed_words is None or len(seed_words) == 0:
            current_words = [random.choice(self.fallback_words)]
        elif isinstance(seed_words, str):
            current_words = seed_words.strip().split()
        else:
            current_words = list(seed_words)
        current_words = [word.lower().strip() for word in current_words if word.strip()]
        if not current_words:
            current_words = [random.choice(self.fallback_words)]
        generated_words = current_words.copy()
        for i in range(length):
            transitions = self.get_multi_word_transitions(current_words)
            if not transitions:
                transitions = self.text_processor.get_transition_probs(current_words[-1])
            if not transitions:
                next_word = random.choice(self.fallback_words)
                generated_words.append(next_word)
                current_words = [next_word]
                continue
            transitions = transitions[:self.max_transitions]
            prob_idx = min(i, len(selection_probs) - 1)
            neural_influence = selection_probs[prob_idx].item()
            context_influence = min(len(current_words) * self.context_weight.item(), 1.0)
            words, weights = zip(*transitions)
            weights = np.array(weights, dtype=float)
            total_influence = 0.5 * neural_influence + 0.3 * context_influence
            weights = weights + total_influence
            if weights.sum() > 0:
                weights = weights / weights.sum()
                next_word = np.random.choice(words, p=weights)
            else:
                next_word = random.choice(words)
            generated_words.append(next_word)
            current_words.append(next_word)
            if len(current_words) > 3:
                current_words = current_words[-3:]
        return ' '.join(generated_words)

# ------------------------------------------------------
# Main SNN Text Generator Class
# ------------------------------------------------------
class SNNTextGenerator:
    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        
        # Initialize components
        self.text_processor = EnhancedTextProcessor(
            num_neurons=settings['hidden_size'],
            device=self.device,
            vocab_limit=settings['vocab_size'],
            max_features=settings['max_features']
        )
        
        self.snn = TrainableStreamingSNN(
            num_neurons=settings['hidden_size'],
            device=self.device
        )
        
        self.generator = TrainableStreamingTextGenerator(
            self.text_processor,
            hidden_dim=settings['hidden_size'] // 2
        )
        
        # Move to device
        self.text_processor.to(self.device)
        self.snn.to(self.device)
        self.generator.to(self.device)
        
        # Optimizer
        all_params = list(self.text_processor.parameters()) + \
                    list(self.snn.parameters()) + \
                    list(self.generator.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=settings['lr'])
        
    def train(self, texts):
        """Train the SNN text generator on provided texts."""
        print("Starting training...")
        
        # Load and process text data
        processed_words = []
        for text in texts:
            if isinstance(text, str):
                words = text.lower().split()
                processed_words.extend(words)
        
        # Build vocabulary and n-gram models
        for i, word in enumerate(processed_words):
            if word not in self.text_processor.word_to_idx:
                if len(self.text_processor.word_to_idx) < self.text_processor.vocab_limit:
                    self.text_processor.word_to_idx[word] = len(self.text_processor.word_to_idx)
            
            # Build n-gram counts
            if i > 0:
                bigram = (processed_words[i-1], word)
                self.text_processor.bigram_counts[bigram] += 1
            if i > 1:
                trigram = (processed_words[i-2], processed_words[i-1], word)
                self.text_processor.trigram_counts[trigram] += 1
        
        # Fit TF-IDF vectorizer
        if not self.text_processor.is_vectorizer_fitted:
            self.text_processor.fit_vectorizer(texts)
        
        # Create training dataset
        dataset = create_dataset(self.text_processor, max_samples=1000)
        
        # Training loop
        self.text_processor.train()
        self.snn.train()
        self.generator.train()
        
        for epoch in range(self.settings['epochs']):
            total_loss = 0
            batch_count = 0
            
            # Reset SNN state at the beginning of each epoch
            self.snn.reset_neurons()
            
            for batch_idx, sequence in enumerate(dataset[:100]):  # Limit for demo
                try:
                    # Clear gradients first
                    self.optimizer.zero_grad()
                    
                    # Convert sequence to neural features
                    if isinstance(sequence, str):
                        words = sequence.split()
                    else:
                        words = sequence
                    
                    # Skip empty sequences
                    if not words:
                        continue
                    
                    # Detach any previous computations to break the graph
                    with torch.no_grad():
                        neural_features = self.text_processor.words_to_neural_features(words)
                    
                    # Enable gradients for current forward pass
                    neural_features = neural_features.detach().requires_grad_(True)
                    
                    # Reset SNN state for each batch to avoid graph accumulation
                    self.snn.reset_neurons()
                    
                    # Process through SNN
                    spikes = self.snn.forward_chunk(neural_features.squeeze(0))
                    
                    # Generate predictions
                    predictions = self.generator(spikes.unsqueeze(0))
                    
                    # Create a more meaningful target based on the input
                    # Fix tensor indexing issues
                    target = torch.zeros_like(predictions)
                    
                    # Ensure we're working with the correct dimensions
                    if predictions.dim() >= 2:
                        batch_size, feature_size = predictions.shape[0], predictions.shape[-1]
                        
                        # Set some neurons to fire based on geometric terms presence
                        for i, word in enumerate(words[:min(len(words), feature_size)]):
                            if word.lower() in self.text_processor.geometric_terms:
                                # Safe indexing - ensure we don't exceed tensor dimensions
                                idx = min(i, feature_size - 1)
                                if batch_size > 0:
                                    target[0, idx] = 1.0
                    else:
                        # Handle 1D case
                        feature_size = predictions.shape[0] if predictions.dim() > 0 else 1
                        for i, word in enumerate(words[:min(len(words), feature_size)]):
                            if word.lower() in self.text_processor.geometric_terms:
                                idx = min(i, feature_size - 1)
                                target[idx] = 1.0
                    
                    loss = F.mse_loss(predictions, target)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(
                        list(self.text_processor.parameters()) + 
                        list(self.snn.parameters()) + 
                        list(self.generator.parameters()), 
                        max_norm=1.0
                    )
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Clear cache periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    # Reset states on error to prevent graph contamination
                    self.snn.reset_neurons()
                    continue
            
            avg_loss = total_loss / max(batch_count, 1)  # Prevent division by zero
            print(f"Epoch {epoch+1}/{self.settings['epochs']}, Loss: {avg_loss:.4f}, Batches: {batch_count}")
            
            # Reset all states between epochs
            self.snn.reset_neurons()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.is_trained = True
        print("Training completed!")
    
    def generate(self, prompt="", length=50):
        """Generate text using the trained model."""
        if not self.is_trained:
            return "Model not trained yet. Please train the model first."
        
        self.text_processor.eval()
        self.snn.eval()
        self.generator.eval()
        
        with torch.no_grad():
            # Process prompt
            if prompt:
                words = prompt.lower().split()
            else:
                words = ["the"]
            
            # Get neural features
            neural_features = self.text_processor.words_to_neural_features(words)
            
            # Process through SNN
            spikes = self.snn.forward_chunk(neural_features.squeeze(0))
            
            # Generate text
            generated_text = self.generator.generate_text_trainable(
                spikes.unsqueeze(0), seed_words=words, length=length
            )
            
            return generated_text
    
    def save_model_and_settings(self, file_path):
        """Saves the entire generator instance to a file."""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model and settings saved to {file_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
    
    @classmethod
    def load_model_and_settings(cls, file_path):
        """Loads a generator instance from a file."""
        try:
            with open(file_path, 'rb') as f:
                generator = pickle.load(f)
            print(f"Model and settings loaded from {file_path}")
            return generator
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None

# ------------------------------------------------------
# Dataset creation and training helper functions
# ------------------------------------------------------
def create_dataset(text_processor, max_samples=1000):
    dataset = []
    sequences = [["theorem", "compass", "only", "constructions"]]
    dataset.extend(sequences)
    word_list = list(text_processor.word_to_idx.keys())
    for i in range(0, min(len(word_list), max_samples//2), 20):
        chunk = word_list[i:i+20]
        dataset.append(chunk)  # Keep as list of words
    print(f"Created dataset with {len(dataset)} samples")
    return dataset

# ------------------------------------------------------
# Settings and GUI Classes
# ------------------------------------------------------
class SettingsPage:
    """A settings window for adjusting model parameters."""
    
    def __init__(self, parent, settings_vars, update_callback):
        self.parent = parent
        self.settings_vars = settings_vars
        self.update_callback = update_callback
        
        self.top = tk.Toplevel(self.parent.root)
        self.top.title("Settings")
        self.top.geometry("400x650")
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.top, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # SNN section
        snn_frame = ttk.LabelFrame(main_frame, text="SNN Parameters", padding=10)
        snn_frame.pack(fill=tk.X, pady=5)
        
        self.add_entry(snn_frame, "Hidden Size:", "hidden_size")
        self.add_entry(snn_frame, "Time Steps:", "time_steps")
        self.add_entry(snn_frame, "Membrane Time Constant (τ_mem):", "tau_mem")
        self.add_entry(snn_frame, "Synaptic Time Constant (τ_syn):", "tau_syn")
        self.add_entry(snn_frame, "Spike Threshold (v_thresh):", "v_thresh")
        self.add_entry(snn_frame, "Reset Potential (v_reset):", "v_reset")
        self.add_entry(snn_frame, "Time Step (dt):", "dt")

        # Text Processor section
        text_frame = ttk.LabelFrame(main_frame, text="Text Processing", padding=10)
        text_frame.pack(fill=tk.X, pady=5)
        
        self.add_entry(text_frame, "Vocabulary Size:", "vocab_size")
        self.add_entry(text_frame, "Max Features (TF-IDF):", "max_features")
        self.add_entry(text_frame, "Min DF:", "min_df")
        self.add_entry(text_frame, "N-gram Range (min):", "ngram_min")
        self.add_entry(text_frame, "N-gram Range (max):", "ngram_max")

        # Training section
        train_frame = ttk.LabelFrame(main_frame, text="Training Parameters", padding=10)
        train_frame.pack(fill=tk.X, pady=5)
        
        self.add_entry(train_frame, "Epochs:", "epochs")
        self.add_entry(train_frame, "Learning Rate:", "lr")
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=self.apply_settings).pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.top.destroy).pack(side=tk.LEFT, expand=True, padx=5)
        
    def add_entry(self, parent_frame, label_text, var_name):
        row_frame = ttk.Frame(parent_frame)
        row_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(row_frame, text=label_text).pack(side=tk.LEFT, padx=5)
        entry = ttk.Entry(row_frame, width=20)
        entry.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        entry.insert(0, str(self.settings_vars[var_name]))
        self.settings_vars[var_name] = entry
        
    def apply_settings(self):
        try:
            new_settings = {
                "hidden_size": int(self.settings_vars["hidden_size"].get()),
                "time_steps": int(self.settings_vars["time_steps"].get()),
                "tau_mem": float(self.settings_vars["tau_mem"].get()),
                "tau_syn": float(self.settings_vars["tau_syn"].get()),
                "v_thresh": float(self.settings_vars["v_thresh"].get()),
                "v_reset": float(self.settings_vars["v_reset"].get()),
                "dt": float(self.settings_vars["dt"].get()),
                "vocab_size": int(self.settings_vars["vocab_size"].get()),
                "max_features": int(self.settings_vars["max_features"].get()),
                "min_df": int(self.settings_vars["min_df"].get()),
                "ngram_min": int(self.settings_vars["ngram_min"].get()),
                "ngram_max": int(self.settings_vars["ngram_max"].get()),
                "epochs": int(self.settings_vars["epochs"].get()),
                "lr": float(self.settings_vars["lr"].get()),
            }
            
            self.update_callback(new_settings)
            self.top.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid numbers: {e}")

class SimpleGUI:
    """Simplified GUI for the SNN text generator."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SNN Text Generator")
        self.root.geometry("800x650")
        
        self.settings = {
            "vocab_size": 5000,
            "max_features": 1000,
            "min_df": 1,
            "ngram_min": 1,
            "ngram_max": 2,
            "hidden_size": 256,
            "epochs": 5,
            "lr": 0.001,
            "time_steps": 10,
            "tau_mem": 10.0,
            "tau_syn": 5.0,
            "v_thresh": 1.0,
            "v_reset": 0.0,
            "dt": 1.0,
        }
        
        self.generator = SNNTextGenerator(self.settings)
        self.message_queue = queue.Queue()
        
        self.create_widgets()
        self.check_queue()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top buttons frame
        top_buttons_frame = ttk.Frame(main_frame)
        top_buttons_frame.pack(fill=tk.X)
        
        # Data loading section
        data_frame = ttk.LabelFrame(top_buttons_frame, text="Data Loading", padding=10)
        data_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(data_frame, text="Load Text File", 
                   command=self.load_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(data_frame, text="Use Sample Data", 
                   command=self.use_sample_data).pack(side=tk.LEFT, padx=5)
        
        self.data_status = ttk.Label(data_frame, text="No data loaded")
        self.data_status.pack(side=tk.LEFT, padx=20)

        # Save/Load buttons
        save_load_frame = ttk.LabelFrame(top_buttons_frame, text="Model and Settings", padding=10)
        save_load_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Button(save_load_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_load_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # Settings button
        ttk.Button(top_buttons_frame, text="Settings", command=self.open_settings).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Training section
        train_frame = ttk.LabelFrame(main_frame, text="Training", padding=10)
        train_frame.pack(fill=tk.X, pady=(10, 10))
        
        ttk.Button(train_frame, text="Start Training", 
                   command=self.start_training).pack(side=tk.LEFT, padx=5)
        
        self.train_status = ttk.Label(train_frame, text="Not trained")
        self.train_status.pack(side=tk.LEFT, padx=20)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(train_frame, variable=self.progress_var)
        self.progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Generation section
        gen_frame = ttk.LabelFrame(main_frame, text="Text Generation", padding=10)
        gen_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input
        ttk.Label(gen_frame, text="Prompt:").pack(anchor=tk.W)
        self.prompt_entry = tk.Entry(gen_frame, width=50)
        self.prompt_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Length control
        length_frame = ttk.Frame(gen_frame)
        length_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(length_frame, text="Length:").pack(side=tk.LEFT)
        self.length_var = tk.IntVar(value=50)
        length_scale = ttk.Scale(length_frame, from_=10, to=600, 
                                 variable=self.length_var, orient=tk.HORIZONTAL)
        length_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.length_label = ttk.Label(length_frame, text="50")
        self.length_label.pack(side=tk.RIGHT)
        
        def update_length_label(*args):
            self.length_label.config(text=str(self.length_var.get()))
        self.length_var.trace("w", update_length_label)
        
        # Generate button
        ttk.Button(gen_frame, text="Generate Text", 
                   command=self.generate_text).pack(pady=(0, 10))
        
        # Output
        ttk.Label(gen_frame, text="Generated Text:").pack(anchor=tk.W)

        # Create a frame for the text widget and scrollbar
        text_frame = ttk.Frame(gen_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # Create text widget and scrollbar in the same frame
        self.output_text = tk.Text(text_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.output_text.yview)

        # Configure the text widget to use the scrollbar
        self.output_text.config(yscrollcommand=scrollbar.set)

        # Pack the scrollbar first (on the right), then the text widget (fills remaining space)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    
    def check_queue(self):
        """Check message queue for updates."""
        try:
            while True:
                msg_type, message = self.message_queue.get_nowait()
                if msg_type == "status":
                    self.train_status.config(text=message)
                elif msg_type == "progress":
                    self.progress_var.set(message)
                elif msg_type == "data_status":
                    self.data_status.config(text=message)
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_queue)
    
    def load_file(self):
        """Load text file for training."""
        filename = filedialog.askopenfilename(
            title="Select text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into sentences/paragraphs
                texts = [line.strip() for line in content.split('\n') 
                         if line.strip() and len(line.strip()) > 20]
                
                self.training_texts = texts[:1000]  # Limit for demo
                self.message_queue.put(("data_status", f"Loaded {len(self.training_texts)} texts"))
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def use_sample_data(self):
        """Use built-in sample data."""
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by the human brain.",
            "Text generation can be accomplished with various methods.",
            "Spiking neural networks model temporal dynamics.",
            "Natural language processing involves understanding text.",
            "Deep learning has revolutionized many fields.",
            "Artificial intelligence continues to advance rapidly.",
            "Computer science combines mathematics and programming.",
            "Language models can generate coherent text.",
        ] * 10  # Repeat for more data
        
        self.training_texts = sample_texts
        self.message_queue.put(("data_status", f"Using {len(sample_texts)} sample texts"))
    
    def start_training(self):
        """Start training in background thread."""
        if not hasattr(self, 'training_texts'):
            messagebox.showerror("Error", "No training data loaded")
            return
        
        def train_worker():
            try:
                self.message_queue.put(("status", "Training..."))
                self.message_queue.put(("progress", 0))
                
                # Train the model
                self.generator.train(self.training_texts)
                
                self.message_queue.put(("status", "Training completed"))
                self.message_queue.put(("progress", 100))
                
            except Exception as e:
                self.message_queue.put(("status", f"Training failed: {e}"))
        
        thread = threading.Thread(target=train_worker)
        thread.daemon = True
        thread.start()
    
    def generate_text(self):
        """Generate text with current prompt."""
        prompt = self.prompt_entry.get()
        length = self.length_var.get()
        
        try:
            start_time = time.time()
            generated_text = self.generator.generate(prompt, length)
            generation_time = time.time() - start_time
            
            # Display result
            result = f"Prompt: {prompt}\n"
            result += f"Length: {length} words\n"
            result += f"Generation time: {generation_time:.2f}s\n"
            result += f"Neural activity: {'Available' if self.generator.is_trained else 'N/A'}\n\n"
            result += f"Generated text:\n{generated_text}\n\n"
            result += "-" * 50 + "\n"
            
            self.output_text.insert(tk.END, result)
            self.output_text.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Generation failed: {e}")
    
    def open_settings(self):
        """Opens the settings window."""
        SettingsPage(self, self.settings, self.apply_settings)

    def apply_settings(self, new_settings):
        """Applies new settings and re-initializes the generator."""
        self.settings.update(new_settings)
        try:
            self.generator = SNNTextGenerator(self.settings)
            self.message_queue.put(("status", "Settings applied. Model re-initialized."))
            self.generator.is_trained = False
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to apply settings: {e}")
    
    def save_model(self):
        """Prompts for a file path and saves the generator instance."""
        if not self.generator.is_trained:
            messagebox.showwarning("Warning", "The model is not trained. Saving an untrained model.")
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="Save Model and Settings"
        )
        if file_path:
            self.generator.save_model_and_settings(file_path)
            self.message_queue.put(("status", f"Model and settings saved."))

    def load_model(self):
        """Prompts for a file path and loads a generator instance."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="Load Model and Settings"
        )
        if file_path:
            loaded_generator = SNNTextGenerator.load_model_and_settings(file_path)
            if loaded_generator:
                self.generator = loaded_generator
                self.settings = self.generator.settings
                self.message_queue.put(("status", f"Model and settings loaded."))
                self.message_queue.put(("data_status", f"Model loaded from {file_path}"))
                self.train_status.config(text="Model loaded" if self.generator.is_trained else "Not trained")
                self.progress_var.set(100 if self.generator.is_trained else 0)

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

def main():
    """Main application entry point."""
    print("Starting SNN Text Generator...")
    
    app = SimpleGUI()
    app.run()

if __name__ == "__main__":
    main()