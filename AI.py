import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
import pickle

class SimpleSNN(nn.Module):
    """Simplified Spiking Neural Network with proper LIF neurons."""
    
    def __init__(self, input_size, hidden_size, output_size, dt=1.0, tau_mem=10.0, tau_syn=5.0, v_thresh=1.0, v_reset=0.0):
        super().__init__()
        self.dt = dt
        self.hidden_size = hidden_size
        
        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # LIF parameters
        self.tau_mem = nn.Parameter(torch.tensor(float(tau_mem)))  # Membrane time constant
        self.tau_syn = nn.Parameter(torch.tensor(float(tau_syn)))  # Synaptic time constant
        self.v_thresh = nn.Parameter(torch.tensor(float(v_thresh)))  # Spike threshold
        self.v_reset = nn.Parameter(torch.tensor(float(v_reset)))  # Reset potential
        
        # State variables
        self.reset_state()
    
    def reset_state(self):
        """Reset neuron states."""
        self.v_mem = None
        self.i_syn = None
        self.spikes = None
    
    def lif_dynamics(self, input_current, v_mem, i_syn):
        """Leaky Integrate-and-Fire neuron dynamics."""
        # Synaptic current decay
        i_syn = i_syn - (self.dt / self.tau_syn) * i_syn + input_current
        
        # Membrane potential update
        v_mem = v_mem - (self.dt / self.tau_mem) * v_mem + i_syn
        
        # Spike generation
        spikes = (v_mem >= self.v_thresh).float()
        
        # Reset after spike
        v_mem = v_mem * (1 - spikes) + self.v_reset * spikes
        
        return v_mem, i_syn, spikes
    
    def forward(self, x, time_steps=10):
        batch_size = x.shape[0]
        
        # Initialize states if needed
        if self.v_mem is None or self.v_mem.shape[0] != batch_size:
            self.v_mem = torch.zeros(batch_size, self.hidden_size, device=x.device)
            self.i_syn = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        spike_outputs = []
        
        # Process input over time
        for t in range(time_steps):
            # Scale input for this time step
            current_input = self.fc1(x) / time_steps
            
            # Update LIF dynamics
            self.v_mem, self.i_syn, spikes = self.lif_dynamics(
                current_input, self.v_mem, self.i_syn
            )
            
            spike_outputs.append(spikes)
        
        # Sum spikes over time and apply output layer
        total_spikes = torch.stack(spike_outputs, dim=0).sum(dim=0)
        output = self.fc2(total_spikes)
        
        return output, total_spikes

class TextProcessor:
    """Simplified text processing with proper TF-IDF integration."""
    
    def __init__(self, vocab_size=5000, max_features=1000, min_df=1, ngram_range=(1,2)):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<UNK>': 0}
        self.idx_to_word = {0: '<UNK>'}
        
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True,
            stop_words='english',
            min_df=min_df,
            ngram_range=ngram_range
        )
        self.is_fitted = False
        
        # N-gram counts for generation
        self.bigram_counts = Counter()
        self.word_counts = Counter()
    
    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        # Keep most common words
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(self.vocab_size - 1)
        
        for i, (word, count) in enumerate(most_common, 1):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
            self.word_counts[word] = count
        
        # Build bigrams
        for text in texts:
            words = text.lower().split()
            for i in range(len(words) - 1):
                if words[i] in self.word_to_idx and words[i+1] in self.word_to_idx:
                    self.bigram_counts[(words[i], words[i+1])] += 1
    
    def fit_vectorizer(self, texts):
        """Fit TF-IDF vectorizer."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def text_to_features(self, text):
        """Convert text to feature vector."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        
        # TF-IDF features
        tfidf_features = self.vectorizer.transform([text]).toarray()[0]
        
        return torch.tensor(tfidf_features, dtype=torch.float32)
    
    def generate_text(self, seed_words, length=50):
        """Simple bigram-based text generation."""
        if not seed_words:
            seed_words = ['the']
        
        generated = list(seed_words)
        current_word = seed_words[-1]
        
        for _ in range(length):
            # Find possible next words
            candidates = []
            for (w1, w2), count in self.bigram_counts.items():
                if w1 == current_word:
                    candidates.extend([w2] * count)
            
            if not candidates:
                # Fall back to most common words
                candidates = list(self.word_counts.keys())[:10]
            
            if candidates:
                next_word = np.random.choice(candidates)
                generated.append(next_word)
                current_word = next_word
            else:
                break
        
        return ' '.join(generated)

class SNNTextGenerator:
    """Main SNN-based text generator."""
    
    def __init__(self, settings):
        self.settings = settings
        self.text_processor = TextProcessor(
            vocab_size=self.settings["vocab_size"],
            max_features=self.settings["max_features"],
            min_df=self.settings["min_df"],
            ngram_range=(self.settings["ngram_min"], self.settings["ngram_max"])
        )
        self.snn = SimpleSNN(
            input_size=self.settings["max_features"],
            hidden_size=self.settings["hidden_size"],
            output_size=self.settings["vocab_size"],
            dt=self.settings["dt"],
            tau_mem=self.settings["tau_mem"],
            tau_syn=self.settings["tau_syn"],
            v_thresh=self.settings["v_thresh"],
            v_reset=self.settings["v_reset"]
        )
        self.optimizer = None
        self.is_trained = False
    
    def prepare_data(self, texts):
        """Prepare data for training."""
        print("Building vocabulary...")
        self.text_processor.build_vocab(texts)
        
        print("Fitting TF-IDF vectorizer...")
        self.text_processor.fit_vectorizer(texts)
        
        print(f"Vocabulary size: {len(self.text_processor.word_to_idx)}")
        print(f"Bigram pairs: {len(self.text_processor.bigram_counts)}")
    
    def train(self, texts):
        """Train the SNN model."""
        if not texts:
            raise ValueError("No training texts provided")
        
        # Re-initialize the model and processor with current settings
        self.text_processor = TextProcessor(
            vocab_size=self.settings["vocab_size"],
            max_features=self.settings["max_features"],
            min_df=self.settings["min_df"],
            ngram_range=(self.settings["ngram_min"], self.settings["ngram_max"])
        )
        self.snn = SimpleSNN(
            input_size=self.settings["max_features"],
            hidden_size=self.settings["hidden_size"],
            output_size=self.settings["vocab_size"],
            dt=self.settings["dt"],
            tau_mem=self.settings["tau_mem"],
            tau_syn=self.settings["tau_syn"],
            v_thresh=self.settings["v_thresh"],
            v_reset=self.settings["v_reset"]
        )
        self.prepare_data(texts)
        self.optimizer = torch.optim.Adam(self.snn.parameters(), lr=self.settings["lr"])
        
        print(f"Starting training for {self.settings['epochs']} epochs...")
        
        for epoch in range(self.settings["epochs"]):
            epoch_loss = 0.0
            samples_processed = 0
            
            for text in texts[:100]:  # Limit for demo
                try:
                    # Get features
                    features = self.text_processor.text_to_features(text)
                    features = features.unsqueeze(0)  # Add batch dimension
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    output, spikes = self.snn(features, time_steps=self.settings["time_steps"])
                    
                    # Simple reconstruction loss
                    target = torch.randn_like(output)  # Placeholder target
                    loss = F.mse_loss(output, target)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.snn.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    samples_processed += 1
                    
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue
            
            avg_loss = epoch_loss / max(samples_processed, 1)
            print(f"Epoch {epoch+1}/{self.settings['epochs']}: Loss = {avg_loss:.4f}")
        
        self.is_trained = True
        print("Training completed!")
    
    def generate(self, prompt="", length=50):
        """Generate text using the trained model."""
        if not self.is_trained:
            print("Model not trained, using simple generation...")
            seed_words = prompt.split() if prompt else ['the']
            return self.text_processor.generate_text(seed_words, length)
        
        try:
            # Use SNN for generation
            if prompt:
                features = self.text_processor.text_to_features(prompt)
                features = features.unsqueeze(0)
                
                with torch.no_grad():
                    self.snn.reset_state()
                    output, spikes = self.snn(features, time_steps=self.settings["time_steps"])
                
                # Use spike activity to influence generation
                spike_influence = torch.mean(spikes).item()
                print(f"Neural activity level: {spike_influence:.3f}")
            
            # Fall back to n-gram generation
            seed_words = prompt.split() if prompt else ['the']
            return self.text_processor.generate_text(seed_words, length)
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Generation failed."
    
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
        self.add_entry(text_frame, "Min Document Frequency (min_df):", "min_df")
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
        self.output_text = tk.Text(gen_frame, height=15, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for output
        scrollbar = ttk.Scrollbar(self.output_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.output_text.yview)
    
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