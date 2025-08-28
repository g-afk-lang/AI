import torch
import torch.nn as nn
import math
import numpy as np
import json
from pathlib import Path

class MathUtils:
    @staticmethod
    def custom_sigmoid(x, clamp_range=(-10, 10)):
        """Improved sigmoid with configurable clamping."""
        x_clamped = torch.clamp(x, *clamp_range)
        return torch.sigmoid(x_clamped)
    
    @staticmethod
    def gumbel_softmax(logits, temperature=0.1, hard=False):
        """Gumbel-Softmax sampling for differentiable discrete sampling."""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y = torch.softmax((logits + gumbel_noise) / temperature, dim=-1)
        
        if hard:
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = y_hard - y.detach() + y
        
        return y

class DeviceManager:
    @staticmethod
    def get_optimal_device():
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    @staticmethod
    def move_to_device(obj, device):
        """Move object to device safely."""
        if hasattr(obj, 'to'):
            return obj.to(device)
        return obj

class ConfigManager:
    """Centralized configuration management."""
    
    DEFAULT_CONFIG = {
        # Neural Network Settings
        'hidden_size': 256,
        'num_layers': 3,
        'dropout_rate': 0.1,
        
        # SNN Settings
        'tau_mem': 10.0,
        'tau_syn': 5.0,
        'v_thresh': 1.0,
        'v_reset': 0.0,
        'spike_grad_scale': 1.0,
        
        # Text Processing
        'vocab_size': 10000,
        'max_features': 1000,
        'sequence_length': 128,
        'min_df': 2,
        'max_df': 0.95,
        
        # Training Settings
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10,
        'weight_decay': 1e-5,
        'gradient_clip': 1.0,
        'num_workers': 0,  # Set to 0 to avoid DataLoader error
        
        # Generation Settings
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 0.9,
        'max_length': 100,
        
        # System Settings
        'device': 'auto',
        'seed': 42,
        'save_frequency': 5,
        'checkpoint_dir': 'checkpoints',
    }
    
    def __init__(self, config_dict=None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_dict:
            self.config.update(config_dict)
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
    
    def update(self, config_dict):
        self.config.update(config_dict)
    
    def to_dict(self):
        return self.config.copy()
    
    def save_to_file(self, filepath):
        """Save configuration to JSON file."""
        try:
            filepath = Path(filepath)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False
    
    def load_from_file(self, filepath):
        """Load configuration from JSON file."""
        try:
            filepath = Path(filepath)
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                self.config.update(loaded_config)
                return True
        except Exception as e:
            print(f"Failed to load config: {e}")
        return False
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GeometricProcessor(nn.Module):
    """Hackable geometric transformation processor."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Learnable geometric parameters
        self.register_parameter('compass_radius', nn.Parameter(torch.tensor(0.5)))
        self.register_parameter('golden_ratio_scale', nn.Parameter(torch.tensor(1.618)))
        self.register_parameter('pi_scale', nn.Parameter(torch.tensor(math.pi)))
        
        # Transformation layers
        hidden_size = config.get('hidden_size', 256)
        geometric_layers = config.get('geometric_layers', 2)
        self.transform_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(geometric_layers)
        ])
    
    def apply_geometric_transform(self, features):
        """Apply learnable geometric transformations."""
        x = features
        for layer in self.transform_layers:
            x = MathUtils.custom_sigmoid(layer(x))
        
        # Apply geometric operations
        geometric_effect = torch.sin(x * self.pi_scale / 4) * torch.cos(x * self.golden_ratio_scale / 6)
        return x * 0.9 + geometric_effect * 0.1

class AdaptiveLIFNeuron(nn.Module):
    """Hackable Leaky Integrate-and-Fire neuron with adaptive parameters."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Learnable LIF parameters
        self.register_parameter('tau_mem', nn.Parameter(torch.tensor(config.get('tau_mem', 10.0))))
        self.register_parameter('tau_syn', nn.Parameter(torch.tensor(config.get('tau_syn', 5.0))))
        self.register_parameter('v_thresh', nn.Parameter(torch.tensor(config.get('v_thresh', 1.0))))
        self.register_parameter('v_reset', nn.Parameter(torch.tensor(config.get('v_reset', 0.0))))
        self.register_parameter('spike_scale', nn.Parameter(torch.tensor(config.get('spike_grad_scale', 1.0))))
        
        # Adaptive components
        self.adaptation = nn.Parameter(torch.zeros(1))
        self.noise_scale = nn.Parameter(torch.tensor(0.01))
    
    def compute_dynamics(self):
        """Compute neuron dynamics parameters."""
        tau_mem = torch.clamp(self.tau_mem, 1.0, 100.0)
        tau_syn = torch.clamp(self.tau_syn, 1.0, 100.0)
        alpha = torch.exp(-1.0 / tau_mem)  # Membrane decay
        beta = torch.exp(-1.0 / tau_syn)   # Synaptic decay
        return alpha, beta
    
    def forward(self, x, state=None, add_noise=True):
        device = x.device
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size, num_neurons = x.shape[0], x.shape[-1]
        
        if state is None:
            v_mem = torch.zeros(batch_size, num_neurons, device=device)
            i_syn = torch.zeros(batch_size, num_neurons, device=device)
        else:
            v_mem, i_syn = state
        
        alpha, beta = self.compute_dynamics()
        
        # Synaptic current dynamics
        i_syn = beta * i_syn + x
        
        # Add optional noise for exploration
        if add_noise and self.training:
            noise = torch.randn_like(v_mem) * self.noise_scale
            i_syn = i_syn + noise
        
        # Membrane potential dynamics
        v_mem = alpha * v_mem + i_syn
        
        # Adaptive threshold
        thresh = self.v_thresh + self.adaptation * v_mem.mean()
        
        # Spike generation
        if self.training:
            # Differentiable spikes using Gumbel-Softmax
            spike_logits = (v_mem - thresh) * self.spike_scale
            spike_probs = torch.stack([torch.zeros_like(spike_logits), spike_logits], dim=-1)
            spikes = MathUtils.gumbel_softmax(spike_probs, hard=True)[..., 1]
        else:
            spikes = (v_mem >= thresh).float()
        
        # Reset mechanism
        v_mem = v_mem * (1 - spikes) + self.v_reset * spikes
        
        return spikes, (v_mem, i_syn)

class ModularSNN(nn.Module):
    """Hackable modular SNN architecture."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_neurons = config.get('hidden_size', 256)
        self.num_layers = config.get('num_layers', 3)
        
        # Input processing
        self.input_projection = nn.Linear(self.num_neurons, self.num_neurons)
        
        # SNN layers
        self.snn_layers = nn.ModuleList([
            AdaptiveLIFNeuron(config) for _ in range(self.num_layers)
        ])
        
        # Inter-layer connections
        self.layer_connections = nn.ModuleList([
            nn.Linear(self.num_neurons, self.num_neurons) 
            for _ in range(self.num_layers - 1)
        ])
        
        # Output processing
        self.output_projection = nn.Linear(self.num_neurons, self.num_neurons)
        
        # Learnable mixing weights
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        
        self.states = [None] * self.num_layers
    
    def reset_states(self):
        """Reset all neuron states."""
        self.states = [None] * self.num_layers
    
    def forward(self, x, return_all_spikes=False):
        # Input processing
        x = MathUtils.custom_sigmoid(self.input_projection(x))
        
        all_spikes = []
        current_input = x
        
        # Process through SNN layers
        for i, (snn_layer, layer_weight) in enumerate(zip(self.snn_layers, self.layer_weights)):
            spikes, self.states[i] = snn_layer(current_input, self.states[i])
            all_spikes.append(spikes * layer_weight)
            
            # Prepare input for next layer
            if i < len(self.layer_connections):
                current_input = MathUtils.custom_sigmoid(self.layer_connections[i](spikes))
        
        # Combine outputs from all layers
        if self.config.get('layer_mixing', True):
            output_spikes = sum(all_spikes)
        else:
            output_spikes = all_spikes[-1]  # Use only last layer
        
        # Final output processing
        output = MathUtils.custom_sigmoid(self.output_projection(output_spikes))
        
        if return_all_spikes:
            return output, all_spikes
        return output

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import numpy as np
import re


class AdvancedTokenizer:
    """Hackable tokenizer with multiple strategies."""
    
    def __init__(self, config):
        self.config = config
        self.vocab_size = config.get('vocab_size', 10000)
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        self.word_counts = Counter()
        
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts."""
        # Count word frequencies
        for text in texts:
            words = self.tokenize(text)
            self.word_counts.update(words)
        
        # Add most frequent words to vocabulary
        most_common = self.word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        for word, count in most_common:
            if count >= min_freq and word not in self.word_to_id:
                word_id = len(self.word_to_id)
                self.word_to_id[word] = word_id
                self.id_to_word[word_id] = word
    
    def tokenize(self, text):
        """Tokenize text with configurable strategy."""
        strategy = self.config.get('tokenization_strategy', 'word')
        
        if strategy == 'word':
            return text.split() #re.findall(r'\b\w+\b', text.lower())
        elif strategy == 'char':
            return list(text.lower())
        elif strategy == 'subword':
            # Simple subword tokenization
            words = text.split() #re.findall(r'\b\w+\b', text.lower())
            subwords = []
            for word in words:
                if len(word) > 6:
                    subwords.extend([word[:3], word[3:]])
                else:
                    subwords.append(word)
            return subwords
        else:
            return text.split()
    
    def encode(self, text):
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return [self.word_to_id.get(token, self.word_to_id['<UNK>']) for token in tokens]
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        tokens = [self.id_to_word.get(token_id, '<UNK>') for token_id in token_ids]
        return ' '.join(tokens)

class MultiModalTextProcessor(nn.Module):
    """Hackable multi-modal text processor."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')
        self.hidden_size = config.get('hidden_size', 256)
        
        # Initialize tokenizer
        self.tokenizer = AdvancedTokenizer(config)
        
        # TF-IDF processor
        self.tfidf_enabled = config.get('use_tfidf', True)
        if self.tfidf_enabled:
            self.vectorizer = TfidfVectorizer(
                max_features=config.get('max_features', 1000),
                ngram_range=(1, config.get('ngram_max', 2)),
                min_df=config.get('min_df', 2),
                max_df=config.get('max_df', 0.95)
            )
            self.tfidf_scaler = StandardScaler()
            self.tfidf_fitted = False
            
            self.tfidf_projection = nn.Sequential(
                nn.Linear(config.get('max_features', 1000), self.hidden_size // 4),
                nn.Dropout(config.get('dropout_rate', 0.1)),
                nn.Linear(self.hidden_size // 4, self.hidden_size // 4)
            )
        
        # Word embeddings
        vocab_size = config.get('vocab_size', 10000)
        self.word_embeddings = nn.Embedding(vocab_size + 10, self.hidden_size // 4)  # +10 for safety
        
        # Positional embeddings
        max_position = config.get('max_position', 1024)
        self.position_embeddings = nn.Embedding(max_position, self.hidden_size // 4)
        
        # Geometric processor
        self.geometric_processor = GeometricProcessor(config)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config.get('dropout_rate', 0.1)),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # N-gram models for language modeling
        self.ngram_models = {
            'bigram': defaultdict(Counter),
            'trigram': defaultdict(Counter),
            'fourgram': defaultdict(Counter)
        }
    
    def fit_tfidf(self, texts):
        """Fit TF-IDF vectorizer."""
        if not self.tfidf_enabled:
            return
        
        processed_texts = []
        for text in texts:
            if isinstance(text, list):
                text = ' '.join(text)
            processed_texts.append(text)
        
        if processed_texts:
            self.vectorizer.fit(processed_texts)
            tfidf_matrix = self.vectorizer.transform(processed_texts)
            self.tfidf_scaler.fit(tfidf_matrix.toarray())
            self.tfidf_fitted = True
    
    def build_ngram_models(self, texts):
        """Build n-gram language models."""
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            
            # Build different n-gram models
            for i in range(len(tokens) - 1):
                # Bigram
                self.ngram_models['bigram'][tokens[i]][tokens[i + 1]] += 1
                
                # Trigram
                if i < len(tokens) - 2:
                    context = (tokens[i], tokens[i + 1])
                    self.ngram_models['trigram'][context][tokens[i + 2]] += 1
                
                # 4-gram
                if i < len(tokens) - 3:
                    context = (tokens[i], tokens[i + 1], tokens[i + 2])
                    self.ngram_models['fourgram'][context][tokens[i + 3]] += 1
    
    def get_tfidf_features(self, text):
        """Extract TF-IDF features."""
        if not self.tfidf_enabled or not self.tfidf_fitted:
            return torch.zeros(1, self.hidden_size // 4, device=self.device)
        
        if isinstance(text, list):
            text = ' '.join(text)
        
        try:
            tfidf_matrix = self.vectorizer.transform([text])
            tfidf_features = self.tfidf_scaler.transform(tfidf_matrix.toarray())
            tfidf_tensor = torch.tensor(tfidf_features, dtype=torch.float32, device=self.device)
            
            return self.tfidf_projection(tfidf_tensor)
        except:
            return torch.zeros(1, self.hidden_size // 4, device=self.device)
    
    def get_word_features(self, tokens, max_length=None):
        """Extract word embedding features."""
        if max_length is None:
            max_length = self.config.get('sequence_length', 128)
        
        # Truncate or pad sequence
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        token_ids = [self.tokenizer.word_to_id.get(token, self.tokenizer.word_to_id['<UNK>']) 
                    for token in tokens]
        
        # Pad sequence
        while len(token_ids) < max_length:
            token_ids.append(self.tokenizer.word_to_id['<PAD>'])
        
        # Ensure we don't exceed embedding size
        token_ids = [min(tid, self.word_embeddings.num_embeddings - 1) for tid in token_ids]
        
        token_tensor = torch.tensor(token_ids, device=self.device).unsqueeze(0)
        word_embeddings = self.word_embeddings(token_tensor)
        
        # Add positional embeddings
        positions = torch.arange(max_length, device=self.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(positions)
        
        combined_embeddings = word_embeddings + position_embeddings
        return combined_embeddings.mean(dim=1)  # Average pooling
    
    def forward(self, text, return_features=False):
        """Process text through all feature extractors."""
        if isinstance(text, str):
            tokens = self.tokenizer.tokenize(text)
        else:
            tokens = text
        
        # Extract different types of features
        tfidf_features = self.get_tfidf_features(tokens)
        word_features = self.get_word_features(tokens)
        
        # Combine features
        if tfidf_features.shape[1] != self.hidden_size // 4:
            # Pad or truncate TF-IDF features
            target_size = self.hidden_size // 4
            if tfidf_features.shape[1] < target_size:
                padding = torch.zeros(tfidf_features.shape[0], 
                                    target_size - tfidf_features.shape[1], 
                                    device=self.device)
                tfidf_features = torch.cat([tfidf_features, padding], dim=1)
            else:
                tfidf_features = tfidf_features[:, :target_size]
        
        # Concatenate all features
        combined_features = torch.cat([
            tfidf_features,
            word_features,
            torch.zeros(1, self.hidden_size // 2, device=self.device)  # Placeholder for future features
        ], dim=1)
        
        # Apply geometric transformations
        processed_features = self.geometric_processor.apply_geometric_transform(combined_features)
        
        # Final feature fusion
        output_features = self.feature_fusion(processed_features)
        
        if return_features:
            return output_features, {
                'tfidf': tfidf_features,
                'word': word_features,
                'geometric': processed_features
            }
        
        return output_features
    
    def get_transition_probs(self, word):
        """Get word transition probabilities."""
        transitions = []
        for next_word, count in self.ngram_models['bigram'][word].items():
            transitions.append((next_word, count))
        return transitions
    
    def get_ngram_transitions(self, context_words, n=3):
        """Get n-gram transition probabilities."""
        if len(context_words) < n - 1:
            return []
        
        context_key = tuple(context_words[-(n-1):])
        transitions = []
        
        if n == 3:
            for next_word, count in self.ngram_models['trigram'][context_key].items():
                transitions.append((next_word, count))
        
        return transitions

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import pickle


def custom_collate_fn(batch):
    """Custom collate function that detaches gradients properly."""
    features = torch.stack([item['features'].detach() for item in batch])
    tokens = [item['tokens'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    return {
        'features': features,
        'tokens': tokens,
        'length': lengths
    }

def save_checkpoint(model, optimizer, scheduler, path, extra_info=None):
    """Save model checkpoint with all training state."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'extra_info': extra_info or {}
    }
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load model checkpoint and restore training state."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    extra_info = checkpoint.get('extra_info', {})
    print(f"✅ Checkpoint loaded from {path}")
    return extra_info

class TextDataset(Dataset):
    """Hackable dataset class for text data."""
    
    def __init__(self, texts, text_processor, config):
        self.texts = texts
        self.text_processor = text_processor
        self.config = config
        self.sequence_length = config.get('sequence_length', 128)
        
        # Preprocess texts
        self.processed_data = []
        for text in texts:
            tokens = text_processor.tokenizer.tokenize(text)
            if len(tokens) >= 2:  # Minimum sequence length
                self.processed_data.append(tokens)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        tokens = self.processed_data[idx]
        
        # Create input-target pairs
        if len(tokens) > self.sequence_length:
            start_idx = np.random.randint(0, len(tokens) - self.sequence_length)
            tokens = tokens[start_idx:start_idx + self.sequence_length]
        
        # Convert to features and ensure no gradients
        with torch.no_grad():
            features = self.text_processor(tokens)
        
        return {
            'features': features.squeeze(0).detach(),  # Ensure detachment
            'tokens': tokens,
            'length': len(tokens)
        }

class TrainingManager:
    """Comprehensive training manager with checkpointing and monitoring."""
    
    def __init__(self, config):
        self.config = config
        self.device = DeviceManager.get_optimal_device()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Paths
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def create_optimizer(self, model):
        """Create optimizer with configurable parameters."""
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.get('epochs', 100)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.get('step_size', 30), gamma=0.1
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            return None
    
    def compute_loss(self, model_output, targets, loss_type='mse'):
        """Compute loss with different strategies."""
        if loss_type == 'mse':
            return nn.MSELoss()(model_output, targets)
        elif loss_type == 'cosine':
            return 1 - nn.CosineSimilarity(dim=-1)(model_output, targets).mean()
        elif loss_type == 'contrastive':
            # Simple contrastive loss
            pos_loss = nn.MSELoss()(model_output, targets)
            neg_targets = torch.roll(targets, shifts=1, dims=0)
            neg_loss = torch.max(torch.tensor(0.0), 1.0 - nn.MSELoss()(model_output, neg_targets))
            return pos_loss + 0.1 * neg_loss
        else:
            return nn.MSELoss()(model_output, targets)
    
    def train_epoch(self, model, dataloader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            
            # Reset model states
            if hasattr(model, 'reset_states'):
                model.reset_states()
            
            # Forward pass
            optimizer.zero_grad()
            
            # Create targets (simple reconstruction task)
            targets = features.clone()
            
            # Model forward pass
            outputs = model(features)
            
            # Compute loss
            loss = self.compute_loss(outputs, targets, self.config.get('loss_type', 'mse'))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config.get('gradient_clip', 1.0)
                )
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.step += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate_epoch(self, model, val_loader):
        """Validation epoch."""
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = features.clone()
                
                if hasattr(model, 'reset_states'):
                    model.reset_states()
                
                outputs = model(features)
                loss = self.compute_loss(outputs, targets, self.config.get('loss_type', 'mse'))
                
                total_val_loss += loss.item()
                num_val_batches += 1
        
        return total_val_loss / num_val_batches if num_val_batches > 0 else 0
    
    def save_checkpoint(self, model, optimizer, scheduler, filename=None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        extra_info = {
            'epoch': self.epoch,
            'step': self.step,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        
        save_checkpoint(model, optimizer, scheduler, checkpoint_path, extra_info)
        return checkpoint_path
    
    def load_checkpoint(self, model, optimizer, scheduler, checkpoint_path):
        """Load training checkpoint."""
        extra_info = load_checkpoint(model, optimizer, scheduler, checkpoint_path, self.device)
        
        # Restore training state
        self.epoch = extra_info.get('epoch', 0)
        self.step = extra_info.get('step', 0)
        self.best_loss = extra_info.get('best_loss', float('inf'))
        self.training_history = extra_info.get('training_history', [])
        
        return extra_info
    
    def train(self, model, train_dataset, val_dataset=None, progress_callback=None):
        """Main training loop."""
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid the error
            collate_fn=custom_collate_fn
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                num_workers=0,
                collate_fn=custom_collate_fn
            )
        
        # Setup optimizer and scheduler
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        
        # Training loop
        epochs = self.config.get('epochs', 100)
        save_frequency = self.config.get('save_frequency', 10)
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            
            # Training phase
            train_loss = self.train_epoch(model, train_loader, optimizer, None)
            
            # Validation phase
            val_loss = 0
            if val_loader:
                val_loss = self.validate_epoch(model, val_loader)
            
            # Update learning rate
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loss > 0 else train_loss)
                else:
                    scheduler.step()
            
            # Record training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Save checkpoint
            if val_loss > 0 and val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(model, optimizer, scheduler, 'best_model.pth')
            
            if epoch % save_frequency == 0:
                self.save_checkpoint(model, optimizer, scheduler)
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch, epochs, train_loss, val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

class DatasetManager:
    """Manage different data sources and preprocessing."""
    
    def __init__(self, config):
        self.config = config
        
    def load_text_file(self, file_path):
        """Load text from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into sentences or paragraphs
        texts = [line.strip() for line in content.split('\n') if line.strip()]
        return texts
    
    def load_huggingface_dataset(self, dataset_name, split='train', text_field='text'):
        """Load dataset from Hugging Face."""
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, split=split)
            texts = [str(item[text_field]) for item in dataset]
            return texts
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")
    
    def create_synthetic_data(self, num_samples=1000):
        """Create synthetic text data for testing."""
        templates = [
            "The geometric construction uses compass and straightedge.",
            "Circle intersections define key points in Euclidean geometry.",
            "Perpendicular bisectors create symmetric constructions.",
            "Angle bisectors divide angles into equal parts.",
            "Midpoint construction requires circle intersections.",
            "Triangle construction follows geometric principles.",
            "Square inscriptions use compass constructions.",
            "Golden ratio appears in geometric constructions."
        ]
        
        variations = [
            "theorem", "proof", "construction", "method", "technique",
            "compass", "straightedge", "ruler", "tool", "instrument",
            "point", "line", "circle", "arc", "intersection",
            "perpendicular", "parallel", "angle", "bisector", "midpoint"
        ]
        
        texts = []
        for i in range(num_samples):
            template = np.random.choice(templates)
            # Add some variation
            words = template.split()
            if len(words) > 3:
                # Replace random words with variations
                for j in range(np.random.randint(1, 3)):
                    idx = np.random.randint(len(words))
                    words[idx] = np.random.choice(variations)
            texts.append(' '.join(words))
        
        return texts

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import json
import datetime
from pathlib import Path
import numpy as np

class ConfigEditor:
    """Advanced configuration editor with validation."""
    
    def __init__(self, parent, config_manager, update_callback):
        self.parent = parent
        self.config_manager = config_manager
        self.update_callback = update_callback
        self.entry_widgets = {}
        
        self.window = tk.Toplevel(parent)
        self.window.title("Configuration Editor")
        self.window.geometry("600x700")
        self.window.resizable(True, True)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create notebook for different config sections
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Neural Network Settings
        self.create_neural_config_tab(notebook)
        
        # Training Settings
        self.create_training_config_tab(notebook)
        
        # Buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Apply", command=self.apply_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_config).pack(side=tk.LEFT, padx=5)
        #ttk.Button(button_frame, text="Load", command=self.load_config).pack(side=tk.LEFT, padx=5)
        #ttk.Button(button_frame, text="Save", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def create_neural_config_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Neural Network")
        
        # Scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Neural network parameters
        self.neural_vars = {}
        
        params = [
            ("Hidden Size", "hidden_size", "int", "Number of hidden neurons", (64, 1024)),
            ("Number of Layers", "num_layers", "int", "Number of neural layers", (1, 10)),
            ("Dropout Rate", "dropout_rate", "float", "Dropout probability", (0.0, 0.9)),
            ("Vocabulary Size", "vocab_size", "int", "Maximum vocabulary size", (1000, 50000)),
            ("Max Features", "max_features", "int", "Maximum TF-IDF features", (100, 5000)),
            ("Sequence Length", "sequence_length", "int", "Maximum sequence length", (32, 1024)),
        ]
        
        for i, (label, key, dtype, help_text, range_val) in enumerate(params):
            self.create_parameter_row(scrollable_frame, label, key, dtype, help_text, range_val, self.neural_vars, i)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
    
    def create_training_config_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Training")
        
        # Scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        self.training_vars = {}
        
        params = [
            ("Batch Size", "batch_size", "int", "Training batch size", (1, 256)),
            ("Learning Rate", "learning_rate", "float", "Initial learning rate", (1e-6, 1e-1)),
            ("Epochs", "epochs", "int", "Number of training epochs", (1, 1000)),
            ("Weight Decay", "weight_decay", "float", "L2 regularization strength", (0.0, 1e-2)),
            ("Gradient Clip", "gradient_clip", "float", "Gradient clipping threshold", (0.0, 10.0)),
            ("Save Frequency", "save_frequency", "int", "Checkpoint save frequency", (1, 100)),
            ("Temperature", "temperature", "float", "Generation temperature", (0.1, 2.0)),
            ("Max Length", "max_length", "int", "Maximum generation length", (10, 1000)),
        ]
        
        for i, (label, key, dtype, help_text, range_val) in enumerate(params):
            self.create_parameter_row(scrollable_frame, label, key, dtype, help_text, range_val, self.training_vars, i)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
    
    def create_parameter_row(self, parent, label, key, dtype, help_text, range_val, var_dict, row):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Label
        ttk.Label(frame, text=f"{label}:", width=20).pack(side=tk.LEFT, padx=5)
        
        # Get current value
        current_value = self.config_manager.get(key, "")
        
        # Input widget
        var = tk.StringVar(value=str(current_value))
        widget = ttk.Entry(frame, textvariable=var, width=15)
        widget.pack(side=tk.LEFT, padx=5)
        var_dict[key] = (var, dtype, range_val)
        
        # Help button
        if help_text:
            help_btn = ttk.Button(frame, text="?", width=2, 
                                command=lambda: messagebox.showinfo("Help", f"{label}\n\n{help_text}"))
            help_btn.pack(side=tk.LEFT, padx=2)
        
        # Range indicator
        if range_val and isinstance(range_val, (list, tuple)) and dtype in ["int", "float"]:
            ttk.Label(frame, text=f"({range_val[0]}-{range_val[1]})", 
                     foreground="gray").pack(side=tk.LEFT, padx=5)
    
    def apply_config(self):
        try:
            # Collect all variables
            all_vars = {}
            all_vars.update(self.neural_vars)
            all_vars.update(self.training_vars)
            
            # Convert and validate
            new_config = {}
            for key, (var, dtype, range_val) in all_vars.items():
                value = var.get()
                
                # Type conversion
                if dtype == "int":
                    value = int(value)
                    if range_val and not (range_val[0] <= value <= range_val[1]):
                        raise ValueError(f"{key} must be between {range_val[0]} and {range_val[1]}")
                elif dtype == "float":
                    value = float(value)
                    if range_val and not (range_val[0] <= value <= range_val[1]):
                        raise ValueError(f"{key} must be between {range_val[0]} and {range_val[1]}")
                
                new_config[key] = value
            
            # Update config manager
            self.config_manager.update(new_config)
            
            # Callback to update main application
            if self.update_callback:
                self.update_callback(new_config)
            
            messagebox.showinfo("Success", "Configuration updated successfully!")
            self.window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply configuration: {e}")
    
    def reset_config(self):
        self.config_manager.config = ConfigManager.DEFAULT_CONFIG.copy()
        messagebox.showinfo("Reset", "Configuration reset to defaults")
        self.window.destroy()
        # Recreate window with default values
        ConfigEditor(self.parent, self.config_manager, self.update_callback)
    
    def load_config(self):
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            if self.config_manager.load_from_file(file_path):
                messagebox.showinfo("Success", f"Configuration loaded from {file_path}")
                self.window.destroy()
                # Recreate window with loaded values
                ConfigEditor(self.parent, self.config_manager, self.update_callback)
            else:
                messagebox.showerror("Error", "Failed to load configuration file")
    
    def save_config(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            if self.config_manager.save_to_file(file_path):
                messagebox.showinfo("Success", f"Configuration saved to {file_path}")
            else:
                messagebox.showerror("Error", "Failed to save configuration file")

class MainApplication:
    """Main GUI application with all features."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced SNN Text Generator")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.device = DeviceManager.get_optimal_device()
        self.config_manager.set('device', str(self.device))
        
        self.text_processor = None
        self.snn_model = None
        self.training_manager = None
        self.dataset_manager = DatasetManager(self.config_manager)
        
        # GUI state
        self.training_data = []
        self.is_training = False
        self.training_thread = None
        self.message_queue = queue.Queue()
        
        # Create GUI
        self.create_widgets()
        self.create_menu()
        
        # Start message queue processing
        self.process_message_queue()
    
    def create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self.new_project)
        file_menu.add_command(label="Load Project", command=self.load_project)
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        #edit_menu = tk.Menu(menubar, tearoff=0)
        #menubar.add_cascade(label="Edit", menu=edit_menu)
        #edit_menu.add_command(label="Configuration", command=self.open_config_editor)
        #edit_menu.add_command(label="Clear All", command=self.clear_all)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_widgets(self):
        """Create main application widgets."""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for data and training
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel for generation and output
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        self.create_data_panel(left_frame)
        self.create_training_panel(left_frame)
        self.create_generation_panel(right_frame)
        self.create_output_panel(right_frame)
    
    def create_data_panel(self, parent):
        """Create data loading and management panel."""
        data_frame = ttk.LabelFrame(parent, text="Data Management", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Data source selection
        source_frame = ttk.Frame(data_frame)
        source_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(source_frame, text="Data Source:").pack(side=tk.LEFT)
        
        self.data_source_var = tk.StringVar(value="file")
        ttk.Radiobutton(source_frame, text="File", variable=self.data_source_var, 
                       value="file").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(source_frame, text="HuggingFace", variable=self.data_source_var, 
                       value="huggingface").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(source_frame, text="Synthetic", variable=self.data_source_var, 
                       value="synthetic").pack(side=tk.LEFT, padx=10)
        
        # File selection
        file_frame = ttk.Frame(data_frame)
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.RIGHT, padx=(5, 0))
        
        # HuggingFace dataset
        hf_frame = ttk.Frame(data_frame)
        hf_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(hf_frame, text="Dataset:").pack(side=tk.LEFT)
        self.hf_dataset_var = tk.StringVar(value="wikitext-2-raw-v1")
        ttk.Entry(hf_frame, textvariable=self.hf_dataset_var, width=20).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(hf_frame, text="Split:").pack(side=tk.LEFT, padx=(10, 0))
        self.hf_split_var = tk.StringVar(value="train")
        ttk.Combobox(hf_frame, textvariable=self.hf_split_var, 
                    values=["train", "test", "validation"], width=10).pack(side=tk.LEFT, padx=5)
        
        # Load data button
        ttk.Button(data_frame, text="Load Data", command=self.load_data).pack(pady=10)
        
        # Data info
        self.data_info_var = tk.StringVar(value="No data loaded")
        ttk.Label(data_frame, textvariable=self.data_info_var, 
                 foreground="blue").pack(pady=(0, 10))
        
        # Data preview
        ttk.Label(data_frame, text="Data Preview:").pack(anchor=tk.W)
        self.data_preview = scrolledtext.ScrolledText(data_frame, height=4, state="disabled")
        self.data_preview.pack(fill=tk.BOTH, expand=True)
    
    def create_training_panel(self, parent):
        """Create training control panel."""
        training_frame = ttk.LabelFrame(parent, text="Training Control", padding=10)
        training_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Training controls
        control_frame = ttk.Frame(training_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.train_button = ttk.Button(control_frame, text="Start Training", 
                                      command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Training", 
                                     command=self.stop_training, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress
        progress_frame = ttk.Frame(training_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.RIGHT)
        
        # Training status
        self.training_status_var = tk.StringVar(value="Ready")
        ttk.Label(training_frame, textvariable=self.training_status_var, 
                 foreground="green").pack(pady=(0, 10))
        
        # Quick training settings
        quick_frame = ttk.LabelFrame(training_frame, text="Quick Settings")
        quick_frame.pack(fill=tk.X)
        
        # Epochs
        epochs_frame = ttk.Frame(quick_frame)
        epochs_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(epochs_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=self.config_manager.get('epochs', 10))
        ttk.Spinbox(epochs_frame, from_=1, to=1000, textvariable=self.epochs_var, 
                   width=10).pack(side=tk.RIGHT)
        
        # Learning rate
        lr_frame = ttk.Frame(quick_frame)
        lr_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_var = tk.DoubleVar(value=self.config_manager.get('learning_rate', 0.001))
        ttk.Entry(lr_frame, textvariable=self.lr_var, width=12).pack(side=tk.RIGHT)
        
        # Batch size
        batch_frame = ttk.Frame(quick_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_var = tk.IntVar(value=self.config_manager.get('batch_size', 32))
        ttk.Spinbox(batch_frame, from_=1, to=256, textvariable=self.batch_var, 
                   width=10).pack(side=tk.RIGHT)
    
    def create_generation_panel(self, parent):
        """Create text generation panel."""
        gen_frame = ttk.LabelFrame(parent, text="Text Generation", padding=10)
        gen_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Generation settings
        settings_frame = ttk.Frame(gen_frame)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Temperature
        temp_frame = ttk.Frame(settings_frame)
        temp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(temp_frame, text="Temperature:").pack(side=tk.LEFT)
        self.temperature_var = tk.DoubleVar(value=self.config_manager.get('temperature', 1.0))
        ttk.Scale(temp_frame, from_=0.1, to=2.0, variable=self.temperature_var, 
                 orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.temp_label = ttk.Label(temp_frame, text="1.0")
        self.temp_label.pack(side=tk.RIGHT)
        
        def update_temp_label(*args):
            self.temp_label.config(text=f"{self.temperature_var.get():.2f}")
        self.temperature_var.trace("w", update_temp_label)
        
        # Max length
        length_frame = ttk.Frame(settings_frame)
        length_frame.pack(fill=tk.X, pady=2)
        ttk.Label(length_frame, text="Max Length:").pack(side=tk.LEFT)
        self.max_length_var = tk.IntVar(value=self.config_manager.get('max_length', 500))
        ttk.Spinbox(length_frame, from_=100, to=1000, textvariable=self.max_length_var, 
                   width=10).pack(side=tk.RIGHT)
        
        # Prompt input
        ttk.Label(gen_frame, text="Prompt:").pack(anchor=tk.W, pady=(10, 0))
        self.prompt_text = scrolledtext.ScrolledText(gen_frame, height=4)
        self.prompt_text.pack(fill=tk.X, pady=(0, 10))
        
        # Generation buttons
        button_frame = ttk.Frame(gen_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="Generate", command=self.generate_text).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear", command=self.clear_generation).pack(side=tk.RIGHT)
    
    def create_output_panel(self, parent):
        """Create output display panel."""
        output_frame = ttk.LabelFrame(parent, text="Generated Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Output text
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Output controls
        control_frame = ttk.Frame(output_frame)
        control_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(control_frame, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Copy to Clipboard", command=self.copy_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear", command=self.clear_output).pack(side=tk.RIGHT)
    
    # Event handlers and functionality methods
    def open_config_editor(self):
        """Open the configuration editor."""
        ConfigEditor(self.root, self.config_manager, self.update_config)
    
    def update_config(self, new_config):
        """Update configuration and reinitialize models if needed."""
        self.config_manager.update(new_config)
        # Update quick settings
        self.epochs_var.set(self.config_manager.get('epochs', 10))
        self.lr_var.set(self.config_manager.get('learning_rate', 0.001))
        self.batch_var.set(self.config_manager.get('batch_size', 32))
        self.temperature_var.set(self.config_manager.get('temperature', 1.0))
        self.max_length_var.set(self.config_manager.get('max_length', 100))
        
        # Reinitialize models if they exist
        if self.text_processor or self.snn_model:
            self.initialize_models()
    
    def initialize_models(self):
        """Initialize or reinitialize the models."""
        try:
            # Initialize text processor
            self.text_processor = MultiModalTextProcessor(self.config_manager)
            self.text_processor.to(self.device)
            
            # Initialize SNN model
            self.snn_model = ModularSNN(self.config_manager)
            self.snn_model.to(self.device)
            
            # Initialize training manager
            self.training_manager = TrainingManager(self.config_manager)
            
            self.message_queue.put(("status", "Models initialized successfully"))
            
        except Exception as e:
            self.message_queue.put(("error", f"Failed to initialize models: {e}"))
    
    def browse_file(self):
        """Browse for data file."""
        file_path = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[
                ("Text files", "*.txt"),
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def load_data(self):
        """Load data based on selected source."""
        try:
            source = self.data_source_var.get()
            
            if source == "file":
                file_path = self.file_path_var.get()
                if not file_path:
                    messagebox.showerror("Error", "Please select a file first")
                    return
                self.training_data = self.dataset_manager.load_text_file(file_path)
                
            elif source == "huggingface":
                dataset_name = self.hf_dataset_var.get()
                split = self.hf_split_var.get()
                if not dataset_name:
                    messagebox.showerror("Error", "Please enter a dataset name")
                    return
                self.training_data = self.dataset_manager.load_huggingface_dataset(
                    dataset_name, split
                )
                
            elif source == "synthetic":
                num_samples = 1000
                self.training_data = self.dataset_manager.create_synthetic_data(num_samples)
            
            # Limit data for demo
            self.training_data = self.training_data[:1000]
            
            # Update info and preview
            self.data_info_var.set(f"Loaded {len(self.training_data)} samples")
            
            # Show preview
            self.data_preview.config(state="normal")
            self.data_preview.delete(1.0, tk.END)
            preview_text = "\n".join(self.training_data[:3])
            if len(preview_text) > 300:
                preview_text = preview_text[:300] + "..."
            self.data_preview.insert(1.0, preview_text)
            self.data_preview.config(state="disabled")
            
            # Initialize models if not already done
            if not self.text_processor:
                self.initialize_models()
            
            # Fit text processor if we have data
            if self.text_processor and self.training_data:
                self.text_processor.tokenizer.build_vocab(self.training_data)
                self.text_processor.fit_tfidf(self.training_data)
                self.text_processor.build_ngram_models(self.training_data)
            
            self.message_queue.put(("status", f"Data loaded: {len(self.training_data)} samples"))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def start_training(self):
        """Start the training process."""
        if not self.training_data:
            messagebox.showerror("Error", "Please load training data first")
            return
        
        if not self.text_processor or not self.snn_model:
            self.initialize_models()
        
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress")
            return
        
        # Update config with quick settings
        self.config_manager.set('epochs', self.epochs_var.get())
        self.config_manager.set('learning_rate', self.lr_var.get())
        self.config_manager.set('batch_size', self.batch_var.get())
        
        # Start training in separate thread
        self.is_training = True
        self.train_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        self.training_thread = threading.Thread(target=self._training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _training_worker(self):
        """Training worker thread."""
        try:
            # Create dataset
            dataset = TextDataset(self.training_data, self.text_processor, self.config_manager)
            
            # Create validation split
            val_size = int(0.1 * len(dataset))
            train_size = len(dataset) - val_size
            
            if train_size > 0 and val_size > 0:
                train_dataset, val_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size]
                )
            else:
                train_dataset = dataset
                val_dataset = None
            
            # Training callback
            def progress_callback(epoch, total_epochs, train_loss, val_loss):
                progress = (epoch + 1) / total_epochs * 100
                self.message_queue.put(("progress", progress))
                self.message_queue.put(("training_update", (epoch, train_loss, val_loss)))
            
            # Start training
            self.training_manager.train(
                self.snn_model, 
                train_dataset, 
                val_dataset, 
                progress_callback
            )
            
            self.message_queue.put(("training_complete", None))
            
        except Exception as e:
            self.message_queue.put(("error", f"Training failed: {e}"))
        finally:
            self.is_training = False
    
    def stop_training(self):
        """Stop the training process."""
        self.is_training = False
        self.train_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.training_status_var.set("Training stopped")
    
    def generate_text(self):
        """Enhanced text generation with better creativity."""
        if not self.text_processor or not self.snn_model:
            messagebox.showerror("Error", "Please load and train a model first")
            return
        
        try:
            prompt = self.prompt_text.get(1.0, tk.END).strip()
            max_length = self.max_length_var.get()
            temperature = self.temperature_var.get()
            
            # Initialize with prompt or random start
            if prompt:
                tokens = self.text_processor.tokenizer.tokenize(prompt)
            else:
                # Start with random common words
                tokens = [np.random.choice(["the", "a", "an", "this", "that", "some", "many", "few"])]
            
            generated_tokens = tokens.copy()
            
            # Enhanced generation with fallback strategies
            for i in range(max_length):
                next_word = None
                
                # Strategy 1: Try trigram
                if len(generated_tokens) >= 2:
                    trigram_transitions = self.text_processor.get_ngram_transitions(generated_tokens[-2:], n=3)
                    if trigram_transitions:
                        words, counts = zip(*trigram_transitions)
                        probs = np.array(counts, dtype=float)
                        probs = probs / probs.sum()
                        
                        # Apply temperature
                        if temperature != 1.0:
                            probs = np.power(probs, 1.0 / temperature)
                            probs = probs / probs.sum()
                        
                        next_word = np.random.choice(words, p=probs)
                
                # Strategy 2: Try bigram
                if not next_word and len(generated_tokens) >= 1:
                    transitions = self.text_processor.get_transition_probs(generated_tokens[-1])
                    if transitions:
                        words, counts = zip(*transitions)
                        probs = np.array(counts, dtype=float)
                        probs = probs / probs.sum()
                        
                        # Apply temperature with randomness boost
                        if temperature != 1.0:
                            probs = np.power(probs, 1.0 / temperature)
                            probs = probs / probs.sum()
                        
                        # Add some randomness to avoid repetition
                        if np.random.random() < 0.3:  # 30% chance of random selection
                            next_word = np.random.choice(words)
                        else:
                            next_word = np.random.choice(words, p=probs)
                
                # Strategy 3: Fallback to common words
                if not next_word:
                    fallback_words = [
                        "and", "the", "of", "to", "a", "in", "is", "it", "you", "that",
                        "he", "was", "for", "on", "are", "as", "with", "his", "they",
                        "i", "at", "be", "this", "have", "from", "or", "one", "had",
                        "by", "word", "but", "not", "what", "all", "were", "we", "when",
                        "your", "can", "said", "there", "each", "which", "she", "do",
                        "how", "their", "if", "will", "up", "other", "about", "out",
                        "many", "then", "them", "these", "so", "some", "her", "would",
                        "make", "like", "into", "him", "time", "has", "two", "more",
                        "very", "after", "words", "first", "where", "much", "through"
                    ]
                    next_word = np.random.choice(fallback_words)
                
                generated_tokens.append(next_word)
                
                # Add some variety by occasionally breaking patterns
                if i > 5 and np.random.random() < 0.1:  # 10% chance to inject creativity
                    creative_words = ["however", "meanwhile", "suddenly", "therefore", "perhaps", "indeed"]
                    if np.random.random() < 0.5:
                        generated_tokens.append(np.random.choice(creative_words))
            
            generated_text = ' '.join(generated_tokens)
            
            # Display results
            self.display_generation_result(prompt, generated_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Generation failed: {e}")

    
    def display_generation_result(self, prompt, generated_text):
        """Display generation results."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Main output
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, f"\n{'='*50}\n")
        self.output_text.insert(tk.END, f"Generated at: {timestamp}\n")
        self.output_text.insert(tk.END, f"Prompt: {prompt}\n")
        self.output_text.insert(tk.END, f"{'='*50}\n")
        self.output_text.insert(tk.END, f"{generated_text}\n")
        self.output_text.see(tk.END)
        self.output_text.config(state="disabled")
    
    def clear_generation(self):
        """Clear generation inputs."""
        self.prompt_text.delete(1.0, tk.END)
    
    def save_output(self):
        """Save generated output to file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Output",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                output_content = self.output_text.get(1.0, tk.END)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(output_content)
                messagebox.showinfo("Success", f"Output saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save output: {e}")
    
    def copy_output(self):
        """Copy output to clipboard."""
        output_content = self.output_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(output_content)
        messagebox.showinfo("Success", "Output copied to clipboard")
    
    def clear_output(self):
        """Clear all output."""
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state="disabled")
    
    def new_project(self):
        """Create new project."""
        if messagebox.askyesno("New Project", "Create a new project? This will clear all current data."):
            self.clear_all()
            self.config_manager = ConfigManager()
            messagebox.showinfo("Success", "New project created")
    
    def load_project(self):
        """Load project from file."""
        file_path = filedialog.askopenfilename(
            title="Load Project",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    project_data = json.load(f)
                
                # Load configuration
                if 'config' in project_data:
                    self.config_manager.update(project_data['config'])
                    self.update_gui_from_config()
                
                messagebox.showinfo("Success", f"Project loaded from {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load project: {e}")
    
    def save_project(self):
        """Save current project to file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                project_data = {
                    'config': self.config_manager.to_dict(),
                    'data_info': self.data_info_var.get(),
                    'training_status': self.training_status_var.get(),
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(file_path, 'w') as f:
                    json.dump(project_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Project saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save project: {e}")
    
    def update_gui_from_config(self):
        """Update GUI elements from loaded configuration."""
        # Update quick settings
        self.epochs_var.set(self.config_manager.get('epochs', 10))
        self.lr_var.set(self.config_manager.get('learning_rate', 0.001))
        self.batch_var.set(self.config_manager.get('batch_size', 32))
        self.temperature_var.set(self.config_manager.get('temperature', 1.0))
        self.max_length_var.set(self.config_manager.get('max_length', 500))
    
    def clear_all(self):
        """Clear all data and reset application."""
        self.training_data = []
        self.data_info_var.set("No data loaded")
        self.training_status_var.set("Ready")
        self.progress_var.set(0)
        
        self.data_preview.config(state="normal")
        self.data_preview.delete(1.0, tk.END)
        self.data_preview.config(state="disabled")
        
        self.clear_output()
        self.clear_generation()
        
        # Reset models
        self.text_processor = None
        self.snn_model = None
        self.training_manager = None
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
Advanced SNN Text Generator

A sophisticated text generation system using Spiking Neural Networks
with configurable parameters and advanced features.

Version: 1.0
        """
        messagebox.showinfo("About", about_text)
    
    def process_message_queue(self):
        """Process messages from background threads."""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == "status":
                    self.training_status_var.set(data)
                elif msg_type == "progress":
                    self.progress_var.set(data)
                    self.progress_label.config(text=f"{data:.1f}%")
                elif msg_type == "training_update":
                    epoch, train_loss, val_loss = data
                    # Update any training monitors here
                elif msg_type == "training_complete":
                    self.train_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.training_status_var.set("Training completed")
                    self.progress_var.set(100)
                    self.progress_label.config(text="100%")
                elif msg_type == "error":
                    messagebox.showerror("Error", data)
                    self.training_status_var.set("Error occurred")
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_message_queue)
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced SNN Text Generator")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    if args.config:
        config_manager.load_from_file(args.config)
    
    if args.no_gui:
        print("Command line interface not implemented yet")
        # Would implement CLI functionality here
    else:
        # GUI interface
        app = MainApplication()
        app.run()

if __name__ == "__main__":
    main()
