import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import random
import math
import time
from pathlib import Path
import json

# Check for Hugging Face datasets availability
try:
    from datasets import load_dataset, concatenate_datasets
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

KB_LEN = 99999

# ------------------------------------------------------
# Utility
# ------------------------------------------------------
def custom_sigmoid(x):
    """Heavy sigmoid function using -5/x formulation with safety handling."""
    x_safe = torch.where(torch.abs(x) > torch.tensor(0.5), x, torch.exp(x) * 1.5)
    return torch.sigmoid(-5.0 / x_safe)

# ------------------------------------------------------
# Math Processor
# ------------------------------------------------------
class MathProcessor(nn.Module):
    """Mathematical processor implementing construction principles."""
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
    
    def circle_circle_intersection(self, center1, radius1, center2, radius2):
        d = torch.norm(center2 - center1)
        intersect_condition = torch.logical_or(d <= (radius1 + radius2), d >= torch.abs(radius1 - radius2))
        if not intersect_condition.any():
            return torch.zeros(2, 2, device=self.device), torch.tensor(False, device=self.device)
        return torch.zeros(2, 2, device=self.device), torch.tensor(True, device=self.device)
        
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
# Enhanced Dataset Manager - NEW
# ------------------------------------------------------
class MultiDatasetManager:
    """Manages multiple Hugging Face datasets with quality verification."""
    
    def __init__(self, text_processor):
        self.text_processor = text_processor
        self.datasets = []
        self.verification_patterns = {
            'question_patterns': ['what', 'how', 'why', 'when', 'where', 'which', 'who'],
            'answer_patterns': ['because', 'due to', 'result', 'therefore', 'since', 'as a result'],
            'explanation_patterns': ['explain', 'describe', 'define', 'meaning', 'concept'],
            'quality_indicators': ['detailed', 'comprehensive', 'step by step', 'example', 'illustration']
        }
        self.dataset_weights = {}
        
    def add_dataset(self, dataset_name, split='train', weight=1.0, text_field=None):
        """Add a Hugging Face dataset to the manager."""
        if not HF_DATASETS_AVAILABLE:
            print("⚠️ Hugging Face datasets library not available")
            return False
            
        try:
            print(f"📥 Loading dataset: {dataset_name}, split: {split}")
            dataset = load_dataset(dataset_name, split=split)
            
            # Auto-detect text field if not specified
            if text_field is None:
                text_fields = ['text', 'content', 'body', 'question', 'answer', 'passage']
                text_field = next((field for field in text_fields if field in dataset.column_names), 
                                dataset.column_names[0])
            
            self.datasets.append({
                'name': dataset_name,
                'data': dataset,
                'text_field': text_field,
                'weight': weight,
                'quality_score': 0.0
            })
            
            self.dataset_weights[dataset_name] = weight
            print(f"✅ Added {dataset_name} with {len(dataset)} samples")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load {dataset_name}: {e}")
            return False
    
    def calculate_quality_score(self, text):
        """Calculate quality score for a text sample."""
        text_lower = text.lower()
        score = 0.0
        
        # Check for question patterns
        question_score = sum(1 for pattern in self.verification_patterns['question_patterns'] 
                           if pattern in text_lower)
        score += question_score * 1.2
        
        # Check for answer patterns  
        answer_score = sum(1 for pattern in self.verification_patterns['answer_patterns']
                         if pattern in text_lower)
        score += answer_score * 1.3
        
        # Check for explanation patterns
        explanation_score = sum(1 for pattern in self.verification_patterns['explanation_patterns']
                              if pattern in text_lower)
        score += explanation_score * 1.25
        
        # Check for quality indicators
        quality_score = sum(1 for pattern in self.verification_patterns['quality_indicators']
                          if pattern in text_lower)
        score += quality_score * 1.15
        
        # Length bonus for detailed responses
        word_count = len(text.split())
        if 50 <= word_count <= 300:
            score += 0.5
        elif word_count > 300:
            score += 0.3
            
        return score 
    
    def get_balanced_samples(self, max_samples_per_dataset=10000):
        """Get balanced samples from all datasets with quality filtering."""
        all_samples = []
        
        for dataset_info in self.datasets:
            dataset_data = dataset_info['data']
            text_field = dataset_info['text_field']
            weight = dataset_info['weight']
            name = dataset_info['name']
            
            samples = []
            quality_scores = []
            
            print(f"🔍 Processing {name} dataset...")
            
            # Handle different dataset structures
            if isinstance(dataset_data, list):
                # Fallback synthetic dataset
                for sample in dataset_data:
                    text = str(sample[text_field])
                    quality_score = self.calculate_quality_score(text)
                    
                    if quality_score >= 1.0:
                        samples.append({
                            'text': text,
                            'quality_score': quality_score,
                            'dataset': name,
                            'weight': weight
                        })
                        quality_scores.append(quality_score)
            else:
                # HuggingFace dataset
                for i, sample in enumerate(dataset_data):
                    if i >= max_samples_per_dataset:
                        break
                        
                    try:
                        text = str(sample[text_field])
                        quality_score = self.calculate_quality_score(text)
                        
                        if quality_score >= 1.0:
                            samples.append({
                                'text': text,
                                'quality_score': quality_score,
                                'dataset': name,
                                'weight': weight
                            })
                            quality_scores.append(quality_score)
                    except (KeyError, TypeError) as e:
                        # Skip samples with missing fields
                        continue
            
            # Update dataset quality score
            dataset_info['quality_score'] = np.mean(quality_scores) if quality_scores else 0.0
            
            print(f"📊 {name}: {len(samples)} quality samples, avg score: {dataset_info['quality_score']:.2f}")
            all_samples.extend(samples)
        
        # Sort by quality score and return top samples
        all_samples.sort(key=lambda x: x['quality_score'], reverse=True)
        return all_samples

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
        self.global_adaptation = nn.Parameter(torch.ones(1) * 0.5)
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
        x_hidden = custom_sigmoid(self.hidden_layer(x_processed) * self.activation_scale2)
        prob_weights = custom_sigmoid(x_hidden)
        x_modulated = torch.logical_or(x_chunk <= (x_hidden + x_processed),
                                     x_chunk >= torch.abs(x_hidden - x_processed)) * prob_weights.unsqueeze(0)
        output = custom_sigmoid(self.output_layer(x_modulated))
        adapted_output = output - self.global_adaptation * (1 + x_hidden)
        return adapted_output.squeeze(0)
    
    def forward(self, x_sequence):
        outputs = []
        self.reset_neurons()
        for x in x_sequence:
            out = self.forward_chunk(x)
            outputs.append(out)
        return torch.vstack(outputs) if outputs else torch.empty(0, self.num_neurons, device=self.device)
    
    def reset_neurons(self):
        self.neuron_state = None

# ------------------------------------------------------
# Enhanced Text Processor with Multi-Dataset Support - ENHANCED
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
        self.dataset_manager = MultiDatasetManager(self)
        # ADD THESE LINES to fix the AttributeError:
        self.datasets = self.dataset_manager.datasets
        self.dataset_weights = self.dataset_manager.dataset_weights
        # Enhanced vectorizer for better generic question handling
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,  # Increased min_df for better generalization
            max_df=0.95,  # Slightly more restrictive
            lowercase=True,
            token_pattern=r'\b[a-zA-Z0-9]+\b',
            stop_words=None  # Keep stop words for question processing
        )
        
        self.tfidf_scaler = StandardScaler()
        self.is_vectorizer_fitted = False
        
        # Enhanced projections
        self.tfidf_projection = nn.Sequential(
            nn.Linear(max_features, num_neurons // 4),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(num_neurons // 4, num_neurons // 4),
            nn.LayerNorm(num_neurons // 4)
        )
        
        self.word_embeddings = nn.Embedding(vocab_limit + 1, num_neurons // 4)
        self.position_embeddings = nn.Embedding(1000, num_neurons // 4)
        self.geometric_embeddings = nn.Embedding(100, num_neurons // 4)
        
        # Question-specific processing
        self.question_processor = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.LayerNorm(num_neurons)
        )
        
        self.compass_feature_processor = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons)
        )
        
        self.register_parameter('geometric_sigmoid_scale', nn.Parameter(torch.tensor(1.2)))
        self.register_parameter('tfidf_sigmoid_scale', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('question_weight', nn.Parameter(torch.tensor(0.3)))
        
        # Enhanced geometric terms
        self.geometric_terms = {
            'compass': 0, 'circle': 1, 'intersection': 2, 'construction': 3,
            'midpoint': 4, 'perpendicular': 5, 'radius': 6, 'center': 7,
            'arc': 8, 'point': 9, 'line': 10, 'geometry': 11,
            'mohr': 12, 'theorem': 13, 'euclidean': 14,
            'straightedge': 15, 'triangle': 16, 'square': 17, 'polygon': 18,
            'angle': 19, 'bisector': 20, 'chord': 21, 'diameter': 22,
            'tangent': 23, 'secant': 24, 'vertex': 25, 'edge': 26
        }
        
        # Question patterns for verification
        self.question_patterns = {
            'what': 0, 'how': 1, 'why': 2, 'when': 3, 'where': 4,
            'which': 5, 'who': 6, 'explain': 7, 'describe': 8,
            'define': 9, 'prove': 10, 'solve': 11, 'calculate': 12
        }
        
        self.transition_cache = {}
        self.cache_limit = 1000
        
    def setup_datasets(self):
        """Setup multiple datasets for enhanced training."""
        print("🔧 Setting up multiple datasets...")
        
        # Try different Q&A datasets with correct configurations
        datasets_to_try = [
            ("squad", "plain_text", "train", "context", 2.0),
        ]
        
        datasets_loaded = 0
        for dataset_config in datasets_to_try:
            if len(dataset_config) == 5:
                dataset_name, config_name, split, text_field, weight = dataset_config
                try:
                    if config_name:
                        success = self.add_dataset_with_config(dataset_name, config_name, split, weight, text_field)
                    else:
                        success = self.dataset_manager.add_dataset(dataset_name, split, weight, text_field)
                        
                    if success:
                        datasets_loaded += 1
                        print(f"✅ Successfully loaded {dataset_name}")
                        
                    if datasets_loaded >= 2:  # Stop after loading 2 datasets to avoid memory issues
                        break
                        
                except Exception as e:
                    print(f"⚠️ Failed to load {dataset_name}: {e}")
                    continue
        
        # Always add fallback dataset for robustness
        if datasets_loaded < 2:
            print("⚠️ Adding fallback datasets for robustness...")
            self.create_fallback_dataset()
            datasets_loaded += 1
        
        print(f"📊 Total datasets loaded: {datasets_loaded}")
    
    def add_dataset_with_config(self, dataset_name, config_name, split='train', weight=1.0, text_field=None):
        """Add a Hugging Face dataset with specific configuration."""
        if not HF_DATASETS_AVAILABLE:
            return False
            
        try:
            print(f"📥 Loading dataset: {dataset_name} (config: {config_name}), split: {split}")
            
            # Handle datasets that might not have the expected split
            try:
                dataset = load_dataset(dataset_name, config_name, split=split)
            except ValueError as e:
                if "train" in str(e) and split == "train":
                    print(f"⚠️ 'train' split not found, trying 'validation'...")
                    dataset = load_dataset(dataset_name, config_name, split="validation")
                else:
                    raise e
            
            # Auto-detect text field if not specified
            if text_field is None:
                text_fields = ['text', 'content', 'body', 'question', 'answer', 'passage', 'context', 'document']
                text_field = next((field for field in text_fields if field in dataset.column_names), 
                                dataset.column_names[0])
            
            # Verify the text field exists
            if text_field not in dataset.column_names:
                print(f"⚠️ Field '{text_field}' not found in {dataset.column_names}. Using first available field.")
                text_field = dataset.column_names[0]
            
            self.dataset_manager.datasets.append({
                'name': f"{dataset_name}_{config_name}",
                'data': dataset,
                'text_field': text_field,
                'weight': weight,
                'quality_score': 0.0
            })

            self.dataset_manager.dataset_weights[f"{dataset_name}_{config_name}"] = weight      
            print(f"✅ Added {dataset_name}_{config_name} with {len(dataset)} samples (field: {text_field})")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load {dataset_name} with config {config_name}: {e}")
            return False
    
    def create_fallback_dataset(self):
        """Create a fallback dataset when HuggingFace datasets fail."""
        print("📝 Creating comprehensive fallback synthetic dataset...")
        
        # Enhanced synthetic Q&A samples with geometric focus
        synthetic_samples = [
            {"text": "What is geometry? Geometry is the branch of mathematics that studies shapes, sizes, relative positions of figures, and the properties of space. It includes concepts like points, lines, angles, surfaces, and solids."},
            {"text": "How do you construct a circle with compass? To construct a circle using a compass, place the pointed end at the desired center point, adjust the compass to your desired radius, then rotate the compass 360 degrees while keeping the point fixed at the center."},
            {"text": "What is a theorem in mathematics? A theorem is a mathematical statement that has been rigorously proven to be true based on previously established statements such as axioms, definitions, and other theorems. Theorems form the foundation of mathematical knowledge."},
            {"text": "Why are geometric constructions important? Geometric constructions are important because they help visualize abstract mathematical concepts, develop spatial reasoning skills, and provide practical applications in engineering, architecture, and design. They also demonstrate the logical structure of geometry."},
            {"text": "Explain compass-only construction principles. Compass-only constructions, also known as Mohr-Mascheroni constructions, use only a compass without a straightedge. The Mohr-Mascheroni theorem proves that any construction possible with both compass and straightedge can be accomplished using only a compass."},
            {"text": "Define perpendicular lines in geometry. Perpendicular lines are two lines that intersect at a right angle, forming four 90-degree angles at their intersection point. This relationship is fundamental in geometry and appears in many constructions and proofs."},
            {"text": "What is the Mohr-Mascheroni theorem? The Mohr-Mascheroni theorem, proven by Lorenzo Mascheroni in 1797, states that any geometric construction that can be performed with a compass and straightedge can also be performed using only a compass. This remarkable result shows the power of compass-only constructions."},
            {"text": "How to find intersection points of two circles? To find intersection points of two circles, you need circles with appropriate radii centered at specific points. The intersection points occur where the circumferences of both circles meet, which can be calculated using the distance between centers and the radii of both circles."},
            {"text": "What is a midpoint and how do you construct it? A midpoint is the exact center point of a line segment, dividing it into two equal parts. To construct it with compass only, draw two circles of equal radius centered at each endpoint of the segment. The intersection points of these circles determine the perpendicular bisector, which passes through the midpoint."},
            {"text": "Explain the concept of angle bisector construction. An angle bisector is a line that divides an angle into two equal parts. To construct it, draw arcs of equal radius from the vertex of the angle, then draw arcs of equal radius from where the first arcs intersect the angle's sides. The line from the vertex through the intersection of these arcs is the angle bisector."},
            {"text": "What are the basic elements of Euclidean geometry? The basic elements of Euclidean geometry include points (locations with no dimension), lines (straight paths extending infinitely in both directions), planes (flat surfaces extending infinitely), angles (formed by two intersecting lines), and various shapes like triangles, circles, and polygons."},
            {"text": "How does compass construction relate to mathematical proofs? Compass constructions provide visual and logical demonstrations of geometric relationships. They serve as constructive proofs, showing that certain geometric objects can actually be created following specific rules. This connects abstract mathematical concepts to concrete, performable actions."}
        ]
        
        # Create synthetic dataset
        self.datasets.append({
            'name': 'enhanced_synthetic_fallback',
            'data': synthetic_samples,
            'text_field': 'text',
            'weight': 1.5,  # Higher weight for quality content
            'quality_score': 0.9  # High quality score for well-crafted content
        })
        
        self.dataset_weights['enhanced_synthetic_fallback'] = 1.5
        print(f"✅ Created enhanced fallback dataset with {len(synthetic_samples)} high-quality samples")
        
    def fit_vectorizer(self, documents):
        print("🔧 Fitting enhanced TF-IDF vectorizer...")
        processed_docs = []
        for doc in documents:
            if isinstance(doc, list):
                doc = ' '.join(doc)
            processed_docs.append(doc)
            
        if not processed_docs:
            print("⚠️ No documents available for vectorizer fitting")
            return
            
        self.vectorizer.fit(processed_docs)
        tfidf_matrix = self.vectorizer.transform(processed_docs)
        self.tfidf_scaler.fit(tfidf_matrix.toarray())
        self.is_vectorizer_fitted = True
        
        print(f"✅ Enhanced vectorizer fitted with {len(self.vectorizer.get_feature_names_out())} features")
        
    def detect_question_type(self, words):
        """Detect if input contains question patterns."""
        text = ' '.join(words).lower()
        question_score = 0.0
        
        for pattern in self.question_patterns:
            if pattern in text:
                question_score += 1.0
                
        # Check for question marks or interrogative structure
        if '?' in text or any(word.endswith('?') for word in words):
            question_score += 2.0
            
        return question_score
    
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
        geometric_transform = torch.sin(features * math.pi / 4) + torch.cos(features * math.pi / 6)
        construction_effect = features + 0.1 * geometric_transform
        return construction_effect
    
    def words_to_neural_features(self, words, max_words=50):
        if len(words) > max_words:
            words = words[-max_words:]
            
        device = self.device
        
        # Enhanced TF-IDF processing
        tfidf_features = self.text_to_tfidf_features(words)
        expected_size = self.tfidf_projection[0].in_features
        
        if tfidf_features.shape[1] != expected_size:
            if tfidf_features.shape[1] < expected_size:
                padding = torch.zeros(tfidf_features.shape[0], expected_size - tfidf_features.shape[1], device=device)
                tfidf_features = torch.cat([tfidf_features, padding], dim=1)
            else:
                tfidf_features = tfidf_features[:, :expected_size]
        
        tfidf_processed = custom_sigmoid(self.tfidf_projection(tfidf_features) * self.tfidf_sigmoid_scale)
        
        # Word embeddings
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
        
        # Position embeddings
        position_indices = torch.arange(min(len(words), 999), device=device)
        pos_embs = self.position_embeddings(position_indices)
        pos_features = pos_embs.mean(dim=0, keepdim=True)
        
        # Geometric features
        geo_features = self.encode_geometric_terms(words)
        
        # Question detection and processing
        question_score = self.detect_question_type(words)
        
        # Combine features
        combined_features = torch.cat([tfidf_processed, word_features, pos_features, geo_features], dim=1)
        
        # Apply question-specific processing if detected
        if question_score > 0:
            question_enhanced = self.question_processor(combined_features)
            combined_features = combined_features + self.question_weight * question_enhanced
        
        # Apply compass processing
        compass_features = custom_sigmoid(self.compass_feature_processor(combined_features) * self.geometric_sigmoid_scale)
        
        # Enhanced logical operations with proper dimension handling
        tfidf_slice = tfidf_features[0].unsqueeze(0)
        pos_sum = pos_features + position_indices.float().mean().unsqueeze(0).unsqueeze(0) * geo_features
        pos_diff = torch.abs(pos_features - position_indices.float().mean().unsqueeze(0).unsqueeze(0))
        
        if tfidf_slice.shape[1] != pos_sum.shape[1]:
            min_dim = min(tfidf_slice.shape[1], pos_sum.shape[1])
            tfidf_slice = tfidf_slice[:, :min_dim]
            pos_sum = pos_sum[:, :min_dim]
            pos_diff = pos_diff[:, :min_dim]
        
        logical_condition = torch.logical_not(tfidf_slice >= pos_diff)
        
        if logical_condition.shape[1] != compass_features.shape[1]:
            if logical_condition.shape[1] < compass_features.shape[1]:
                padding = torch.ones(logical_condition.shape[0], 
                                   compass_features.shape[1] - logical_condition.shape[1], 
                                   device=device, dtype=torch.bool)
                logical_condition = torch.cat([logical_condition, padding], dim=1)
            else:
                logical_condition = logical_condition[:, :compass_features.shape[1]]
        
        modified_features = logical_condition.float() * compass_features
        final_features = self.apply_compass_construction_to_features(modified_features)
        
        return final_features

    def load_and_process_multi_dataset(self, max_samples=50000):
        """Load and process multiple Hugging Face datasets."""
        print("🚀 Loading multiple datasets for enhanced training...")
        
        self.setup_datasets()
        quality_samples = self.dataset_manager.get_balanced_samples(max_samples_per_dataset=max_samples//3)
        
        words_processed = []
        documents = []
        current_doc = []
        word_count = 0
        
        # Initialize vocabulary with geometric terms
        vocab = list(self.geometric_terms.keys()) + list(self.question_patterns.keys())
        for word in vocab:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        print(f"📚 Processing {len(quality_samples)} quality samples...")
        
        for sample in quality_samples:
            if KB_LEN > 0 and word_count >= KB_LEN:
                break
                
            text = sample['text']
            words = text.lower().split()
            word_history = []
            
            for word in words:
                if len(self.word_to_idx) < self.vocab_limit:
                    if word not in self.word_to_idx:
                        self.word_to_idx[word] = len(self.word_to_idx)
                
                word_history.append(word)
                
                # Build n-gram models
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
        
        if documents and not self.is_vectorizer_fitted:
            self.fit_vectorizer(documents)
        
        print(f"📚 Processed {word_count} words with vocab size {len(self.word_to_idx)}")
        print(f"📊 Created {len(documents)} documents from {len(quality_samples)} quality samples")
        
        # Print dataset statistics
        for dataset_info in self.dataset_manager.datasets:
            print(f"📈 {dataset_info['name']}: Quality score {dataset_info['quality_score']:.2f}")
        
        return words_processed[-1000:] if words_processed else []
    
    def load_and_process_text_streaming(self, file_path="test.txt", chunk_size=1000, dataset_name=None, split=None):
        """Enhanced version that supports multi-dataset loading."""
        if dataset_name == "multi":
            return self.load_and_process_multi_dataset()
        
        # Original single dataset/file loading logic
        word_count = 0
        documents = []
        current_doc = []
        vocab = list(self.geometric_terms.keys()) + list(self.question_patterns.keys())
        
        for word in vocab:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        words_processed = []
        
        if dataset_name is not None:
            if not HF_DATASETS_AVAILABLE:
                print("⚠️ Hugging Face datasets library not available. Install with: pip install datasets")
                print("⚠️ Falling back to local file loading...")
                return self.load_and_process_text_streaming(file_path=file_path, chunk_size=chunk_size)
            
            try:
                print(f"📥 Loading Hugging Face dataset: {dataset_name}, split: {split}")
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
                print(f"⚠️ Dataset loading failed: {e}. Falling back to local file.")
                return self.load_and_process_text_streaming(file_path=file_path, chunk_size=chunk_size)
        else:
            # Original local file loading logic
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    word_history = []
                    while KB_LEN < 0 or word_count < KB_LEN:
                        chunk = f.read(chunk_size * 10)
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
                print(f"⚠️ File {file_path} not found. Using sample data...")
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
        
        print(f"📚 Processed {word_count} words with vocab size {len(self.word_to_idx)}")
        print(f"📊 Created {len(documents)} documents")
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
# Enhanced Text Generator with Question Verification - ENHANCED
# ------------------------------------------------------
class TrainableStreamingTextGenerator(nn.Module):
    def __init__(self, text_processor, hidden_dim=128, max_transitions_per_word=50):
        super().__init__()
        self.text_processor = text_processor
        self.max_transitions = max_transitions_per_word
        self.fallback_words = ["the", "and", "to", "of", "a", "in", "is", "it", "you", "that"]
        
        # Enhanced selection network
        self.selection_network = nn.Sequential(
            nn.Linear(text_processor.num_neurons, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Question verification network
        self.question_verifier = nn.Sequential(
            nn.Linear(text_processor.num_neurons, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 3)  # [not_question, generic_question, specific_question]
        )
        
        # Answer quality controller
        self.answer_controller = nn.Sequential(
            nn.Linear(text_processor.num_neurons + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.register_parameter('selection_sigmoid_scale', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('verification_scale', nn.Parameter(torch.tensor(0.8)))
        self.context_weight = nn.Parameter(torch.tensor(0.3))
        self.register_parameter('v_thresh', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('quality_threshold', nn.Parameter(torch.tensor(0.6)))
        
        # Quality verification patterns
        self.quality_patterns = {
            'complete_answers': ['because', 'due to', 'therefore', 'as a result', 'consequently'],
            'explanatory_terms': ['step by step', 'first', 'second', 'finally', 'example', 'for instance'],
            'geometric_proofs': ['given', 'prove', 'theorem', 'construction', 'qed', 'therefore'],
            'verification_terms': ['verify', 'check', 'confirm', 'validate', 'ensure']
        }
        
    def verify_answer_quality(self, generated_text, original_question=None):
        """Verify the quality of generated answers."""
        text_lower = generated_text.lower()
        quality_score = 0.0
        
        # Check for complete answer patterns
        for pattern in self.quality_patterns['complete_answers']:
            if pattern in text_lower:
                quality_score += 0.3
        
        # Check for explanatory structure
        for pattern in self.quality_patterns['explanatory_terms']:
            if pattern in text_lower:
                quality_score += 0.2
        
        # Check for geometric proof elements
        for pattern in self.quality_patterns['geometric_proofs']:
            if pattern in text_lower:
                quality_score += 0.25
        
        # Length appropriateness (not too short, not too long)
        word_count = len(generated_text.split())
        if 20 <= word_count <= 200:
            quality_score += 0.3
        elif word_count < 10:
            quality_score -= 0.5  # Penalize very short answers
        
        # Check if answer addresses the question type
        if original_question:
            question_words = set(original_question.lower().split())
            answer_words = set(text_lower.split())
            overlap = len(question_words.intersection(answer_words))
            if overlap > 0:
                quality_score += min(0.2, overlap * 0.05)
        
        return min(quality_score, 1.0)
    
    def classify_question_type(self, spk_rec):
        """Classify input as not_question, generic_question, or specific_question."""
        if spk_rec.numel() == 0:
            return torch.tensor([1.0, 0.0, 0.0], device=next(self.parameters()).device)
        
        with torch.no_grad():
            # Use mean of spike record for classification
            mean_spikes = spk_rec.mean(dim=0) if spk_rec.dim() > 1 else spk_rec
            classification = self.question_verifier(mean_spikes)
            return torch.softmax(classification * self.verification_scale, dim=-1)
    
    def forward(self, spk_rec):
        if spk_rec.numel() == 0:
            return torch.zeros(1, device=next(self.parameters()).device)
        
        # Get question classification
        question_type = self.classify_question_type(spk_rec)
        
        # Base selection network
        linear_output = self.selection_network(spk_rec)
        selection_probs = custom_sigmoid(linear_output.squeeze(-1) * self.selection_sigmoid_scale)
        
        # Enhanced control based on question type
        if spk_rec.dim() > 1:
            combined_input = torch.cat([spk_rec.mean(dim=0).unsqueeze(0), question_type.unsqueeze(0)], dim=1)
        else:
            combined_input = torch.cat([spk_rec.unsqueeze(0), question_type.unsqueeze(0)], dim=1)
        
        quality_control = self.answer_controller(combined_input)
        enhanced_selection = selection_probs * torch.sigmoid(quality_control.squeeze(-1))
        
        return enhanced_selection
    
    def get_multi_word_transitions(self, seed_words):
        if not seed_words:
            return []
        trigram_transitions = self.text_processor.get_ngram_transitions(seed_words, n=3)
        if trigram_transitions:
            return trigram_transitions
        bigram_transitions = self.text_processor.get_transition_probs(seed_words[-1])
        return bigram_transitions
    
    def generate_verified_response(self, spk_rec, seed_words=None, length=50, max_attempts=3):
        """Generate response with quality verification and retry mechanism."""
        best_response = ""
        best_quality = 0.0
        
        for attempt in range(max_attempts):
            response = self.generate_text_trainable(spk_rec, seed_words, length)
            quality = self.verify_answer_quality(response, ' '.join(seed_words) if seed_words else None)
            
            if quality > best_quality:
                best_quality = quality
                best_response = response
            
            # If quality is good enough, return early
            if quality >= self.quality_threshold.item():
                break
            
            # Adjust length for next attempt
            length = min(length + 20, 100)
        
        return best_response, best_quality
    
    def generate_text_trainable(self, spk_rec, seed_words=None, length=50):
        if spk_rec.numel() == 0:
            return "No neural data available for generation."
        
        with torch.no_grad():
            selection_probs = self.forward(spk_rec)
            question_classification = self.classify_question_type(spk_rec)
        
        # Determine if this is a question and adjust generation strategy
        is_question = question_classification[1] + question_classification[2] > 0.1
        
        if seed_words is None or len(seed_words) == 0:
            if is_question:
                current_words = ["to", "answer"]  # Better start for answers
            else:
                current_words = [random.choice(self.fallback_words)]
        elif isinstance(seed_words, str):
            current_words = seed_words.strip().split()
        else:
            current_words = list(seed_words)
        
        current_words = [word.lower().strip() for word in current_words if word.strip()]
        if not current_words:
            current_words = [random.choice(self.fallback_words)]
        
        generated_words = current_words.copy()
        
        # Initialize LIF-like computation
        if len(spk_rec) > 0:
            v_mem = torch.zeros_like(spk_rec[0])
            i_syn = torch.zeros_like(spk_rec[0])
        else:
            v_mem = torch.tensor(0.0, device=self.device)
            i_syn = torch.tensor(0.0, device=self.device)
        
        for i in range(length):
            transitions = self.get_multi_word_transitions(current_words)
            if not transitions:
                transitions = self.text_processor.get_transition_probs(current_words[-1])
            if not transitions:
                # Enhanced fallback for questions
                if is_question and i < length // 2:
                    fallback_words = ['the', 'construction', 'involves', 'using', 'compass', 'to', 'create']
                    next_word = random.choice(fallback_words)
                else:
                    next_word = random.choice(self.fallback_words)
                generated_words.append(next_word)
                current_words = [next_word]
                continue
            
            transitions = transitions[:self.max_transitions]
            
            # Enhanced LIF computation
            beta, alpha = 0.95, 0.1  # Adjusted for better dynamics
            prob_idx = min(i, len(spk_rec) - 1)
            x = spk_rec[prob_idx]
            
            i_syn = alpha * i_syn * x
            membrane_update = i_syn * custom_sigmoid(v_mem)
            v_mem = beta * v_mem + membrane_update
            
            thresh_clamped = self.v_thresh.clamp(0.1, 5.0) * torch.ones_like(v_mem)
            spike_input = (v_mem - thresh_clamped)
            spike_prob = custom_sigmoid(spike_input)
            
            # Gumbel-softmax for discrete selection
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(spike_prob) + 1e-8) + 1e-8)
            spikes = torch.sigmoid((torch.log(spike_prob + 1e-8) - torch.log(1 - spike_prob + 1e-8) + gumbel_noise) / 0.1)
           
            reset_clamped = torch.clamp(spikes, -2.0, 2.0)
            reset_strength = custom_sigmoid(spikes * 5.0)
            v_mem = v_mem * (1 - reset_strength) + reset_clamped * reset_strength
            
            # Enhanced word selection with question awareness
            prob_idx = min(i, len(selection_probs) - 1)
            neural_influence = selection_probs[prob_idx].item()
            context_influence = torch.min(v_mem).item()
            question_influence = 0.3 if is_question else 0.0
            
            words, weights = zip(*transitions)
            weights = np.array(weights, dtype=float)
            
            # Boost weights for question-relevant terms
            if is_question:
                for j, word in enumerate(words):
                    if word in self.quality_patterns['complete_answers']:
                        weights[j] *= 1.5
                    elif word in self.quality_patterns['explanatory_terms']:
                        weights[j] *= 1.3
                    elif word in self.text_processor.geometric_terms:
                        weights[j] *= 1.2
            
            total_influence = 0.5 + neural_influence + context_influence + question_influence
            weights = weights * total_influence
            
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
# Enhanced Dataset Creation with Verification - ENHANCED
# ------------------------------------------------------
def create_enhanced_dataset(text_processor, max_samples=1000000):
    """Create dataset with enhanced verification patterns."""
    dataset = []
    
    # Question-answer pairs for verification
    qa_pairs = [
        ["what is the meaning of life", "the meaning of life involves finding purpose through relationships, personal growth, contribution to society, and the pursuit of knowledge and fulfillment"],
        ["what is consciousness", "consciousness is the subjective experience of being aware, including thoughts, feelings, perceptions, and the sense of self that emerges from brain activity"],
        ["do we have free will", "free will is the ability to make genuine choices, though philosophers debate whether our decisions are truly free or determined by prior causes and brain states"],
        ["what is the nature of reality", "reality consists of the fundamental structure of existence, which may include physical matter, energy, space, time, and possibly non-physical aspects like consciousness or abstract concepts"],
        ["how do we know what is true", "truth is determined through various methods including empirical observation, logical reasoning, reliable testimony, and coherent systems of knowledge that correspond to reality"],
        ["what makes an action morally right", "moral rightness depends on factors like consequences for wellbeing, adherence to universal principles, virtuous character, and respect for human dignity and rights"],
        ["what is the self", "the self is the continuous identity and consciousness that persists through time, encompassing memories, personality, beliefs, and the subjective experience of being you"],
        ["why does anything exist rather than nothing", "existence may be explained by necessary being, quantum fluctuations, infinite regress, or the logical impossibility of absolute nothingness"],
        ["what is the relationship between mind and body", "the mind-body relationship involves how consciousness relates to physical brain processes, whether through identity, emergence, interaction, or fundamental duality"],
        ["what constitutes a good life", "a good life typically involves meaningful relationships, personal fulfillment, moral virtue, health, knowledge, creative expression, and contribution to something greater than oneself"]
    ]

    
    # Add Q&A pairs
    for qa in qa_pairs:
        dataset.append(qa)
    
    # Add vocabulary chunks
    word_list = list(text_processor.word_to_idx.keys())
    for i in range(0, min(len(word_list), max_samples//3), 25):
        chunk = word_list[i:i+25]
        dataset.append(chunk)
    
    print(f"📐 Created enhanced dataset with {len(dataset)} samples")
    print(f"❓ Including {len(qa_pairs)} Q&A pairs")
    
    return dataset

def train_snn_system(text_processor, snn_model, text_generator, dataset, epochs=5, lr=0.001, device='cpu'):
    """Enhanced training with verification feedback."""
    print(f"🎓 Starting enhanced training for {epochs} epochs...")
    
    # Get unique parameters to avoid overlap
    text_gen_params = set(text_generator.parameters())
    text_proc_params = set(text_processor.parameters()) - text_gen_params
    snn_params = set(snn_model.parameters()) - text_gen_params - text_proc_params
    
    # Create optimizer with non-overlapping parameter groups
    param_groups = []
    if text_gen_params:
        param_groups.append({'params': list(text_gen_params), 'lr': lr})
    if text_proc_params:
        param_groups.append({'params': list(text_proc_params), 'lr': lr * 0.5})
    if snn_params:
        param_groups.append({'params': list(snn_params), 'lr': lr * 0.3})
    
    if not param_groups:
        print("⚠️ No parameters found for optimization. Skipping training.")
        return [0.5] * epochs
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    quality_scores = []
    
    for epoch in range(epochs):
        epoch_quality = 0.0
        samples_processed = 0
        total_loss = 0.0
        
        for batch_idx, sample in enumerate(dataset[:100]):  # Limited for demonstration
            if isinstance(sample, list):
                words = sample
            else:
                words = sample.split()
            
            try:
                optimizer.zero_grad()
                
                features = text_processor.words_to_neural_features(words)
                spike_outputs = snn_model.forward(features)
                
                # Generate and verify response
                response, quality = text_generator.generate_verified_response(
                    spike_outputs, seed_words=words[:3], length=30
                )
                
                epoch_quality += quality
                samples_processed += 1
                
                # Create a differentiable quality loss
                # Use selection network output as proxy for quality
                if spike_outputs.numel() > 0:
                    selection_output = text_generator.forward(spike_outputs)
                    target_quality = torch.tensor(quality, device=device, requires_grad=False)
                    
                    # Mean selection probability as quality proxy
                    predicted_quality = selection_output.mean()
                    quality_loss = F.mse_loss(predicted_quality, target_quality)
                    
                    # Add regularization
                    l2_reg = 0.0
                    for param in text_generator.parameters():
                        l2_reg += torch.norm(param)
                    quality_loss += 0.001 * l2_reg
                    
                    quality_loss.backward()
                    torch.nn.utils.clip_grad_norm_(text_generator.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += quality_loss.item()
                
            except Exception as e:
                print(f"⚠️ Training error on sample {batch_idx}: {e}")
                continue
        
        scheduler.step()
        avg_quality = epoch_quality / max(samples_processed, 1)
        avg_loss = total_loss / max(samples_processed, 1)
        quality_scores.append(avg_quality)
        
        print(f"📈 Epoch {epoch+1}/{epochs}: Avg Quality = {avg_quality:.3f}, Loss = {avg_loss:.4f}")
    
    print(f"✅ Training complete. Final quality: {quality_scores[-1]:.3f}")
    return quality_scores

# ------------------------------------------------------
# Enhanced Main Implementation - ENHANCED
# ------------------------------------------------------
def main_implementation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Enhanced model configuration
    num_neurons = 128
    chunk_size = 16
    vocab_limit = 500000  # Increased for multi-dataset
    max_features = 500    # Increased for better feature extraction
    
    # Initialize enhanced models
    text_processor = EnhancedTextProcessor(
        num_neurons, device=device, 
        vocab_limit=vocab_limit, 
        max_features=max_features
    ).to(device)
    
    snn_model = TrainableStreamingSNN(
        num_neurons, device=device, 
        chunk_size=chunk_size
    ).to(device)
    
    text_generator = TrainableStreamingTextGenerator(
        text_processor, hidden_dim=256
    ).to(device)
    
    print("="*70)
    print("🚀 ENHANCED MULTI-DATASET SNN TEXT GENERATOR")
    print("="*70)
    
    # Enhanced dataset loading options
    print("Choose enhanced data source:")
    print("1. Multiple Hugging Face datasets (Recommended)")
    print("2. Local file")
    
    choice = input("Enter choice (1 or 2, default is 1): ").strip() or "1"
    
    words = []
    
    if choice == "1":
        print("🎯 Loading multiple high-quality datasets...")
        words = text_processor.load_and_process_text_streaming(dataset_name="multi")
    
    if choice == "2":
        filename = input("Enter local filename (press Enter for sample_data.txt): ").strip() or "sample_data.txt"
        words = text_processor.load_and_process_text_streaming(file_path=filename)
    
    # Create enhanced dataset
    dataset = create_enhanced_dataset(text_processor)
    
    # Enhanced training
    print("\n🎓 Starting enhanced training with verification...")
    quality_scores = train_snn_system(
        text_processor, snn_model, text_generator, 
        dataset, epochs=2, lr=0.001, device=device
    )
    
    print(f"\n📊 Training Results:")
    print(f"🎯 Final Quality Score: {quality_scores[-1]:.3f}")
    print(f"📈 Quality Improvement: {quality_scores[-1] - quality_scores[0]:.3f}")
    
    print("\n🤖 Interactive Mode with Enhanced Verification:")
    print("💡 Try asking questions like:")
    print("   - 'What is a compass construction?'")
    print("   - 'How do you construct a perpendicular bisector?'")
    print("   - 'Explain the Mohr-Mascheroni theorem'")
    print("   - 'Why are geometric constructions important?'")
    
    while True:
        user_input = input("\n🔸 USER: ").strip()
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        seed_words = user_input.split()
        
        try:
            # Generate enhanced features
            features = text_processor.words_to_neural_features(seed_words)
            spike_outputs = snn_model.forward(features)
            
            # Generate verified response
            response, quality = text_generator.generate_verified_response(
                spike_outputs, seed_words=seed_words, length=500, max_attempts=3
            )
            
            # Display results with quality metrics
            print(f"🤖 AI: {response}")
            print(f"📊 Quality Score: {quality:.2f}/1.0")
            
            if quality < 0.5:
                print("⚠️  Low quality response detected. Trying alternative generation...")
                # Alternative generation attempt
                alt_response = text_generator.generate_text_trainable(
                    spike_outputs, seed_words=seed_words, length=500
                )
                alt_quality = text_generator.verify_answer_quality(alt_response, user_input)
                if alt_quality > quality:
                    print(f"🔄 Alternative: {alt_response}")
                    print(f"📊 Alternative Quality: {alt_quality:.2f}/1.0")
            
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            print("🤖 AI: I apologize, but I encountered an error processing your request.")

if __name__ == "__main__":
    main_implementation()