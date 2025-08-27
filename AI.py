import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk, messagebox, filedialog, simpledialog
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import random
import math
import time
from pathlib import Path
import json
import csv

# Global variables

Filename = "xaa"

KB_LEN = -1  # Set to positive number to limit processing
HF_DATASETS_AVAILABLE = False

# Try to import datasets library
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️ Hugging Face datasets library not available. Install with: pip install datasets")

def custom_sigmoid(x):
    """Heavy sigmoid function using -5/x formulation with safety handling."""
    x_safe = torch.where(torch.abs(x) > torch.tensor(0.5), x, torch.exp(x) * 1.5)
    return torch.sigmoid(-5.0 / x_safe)

# ------------------------------------------------------
# Dataset Load Dialog Class
# ------------------------------------------------------
class DatasetLoadDialog:
    """Dialog for dataset loading options."""
    
    def __init__(self, parent, filename):
        self.result = None
        self.filename = filename
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("📂 Dataset Loading Options")
        self.dialog.geometry("450x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create dialog widgets."""
        # Title
        title_label = tk.Label(self.dialog, text="📂 Load Dataset File", 
                              font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # File info
        file_frame = tk.LabelFrame(self.dialog, text="File Information")
        file_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(file_frame, text=f"File: {Path(self.filename).name}", 
                font=('Arial', 10), wraplength=400).pack(anchor='w', padx=10, pady=5)
        tk.Label(file_frame, text=f"Path: {self.filename}", 
                font=('Arial', 9), fg='#666', wraplength=400).pack(anchor='w', padx=10, pady=(0,5))
        
        # Loading options
        options_frame = tk.LabelFrame(self.dialog, text="Loading Options")
        options_frame.pack(fill='x', padx=20, pady=10)
        
        # Chunk size
        chunk_frame = tk.Frame(options_frame)
        chunk_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(chunk_frame, text="Chunk size:").pack(side=tk.LEFT)
        self.chunk_var = tk.IntVar(value=1000)
        chunk_spinbox = tk.Spinbox(chunk_frame, from_=100, to=10000, increment=100,
                                  textvariable=self.chunk_var)
        chunk_spinbox.pack(side=tk.RIGHT)
        
        # Processing options
        self.replace_dataset_var = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Replace current dataset (clear existing data)",
                      variable=self.replace_dataset_var).pack(anchor='w', padx=10, pady=2)
        
        self.fit_vectorizer_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Refit TF-IDF vectorizer after loading",
                      variable=self.fit_vectorizer_var).pack(anchor='w', padx=10, pady=2)
        
        self.update_models_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Update models after loading",
                      variable=self.update_models_var).pack(anchor='w', padx=10, pady=2)
        
        # File type detection
        format_frame = tk.LabelFrame(self.dialog, text="File Format")
        format_frame.pack(fill='x', padx=20, pady=10)
        
        self.format_var = tk.StringVar(value="auto")
        tk.Radiobutton(format_frame, text="Auto-detect", 
                      variable=self.format_var, value="auto").pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(format_frame, text="Plain text", 
                      variable=self.format_var, value="txt").pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(format_frame, text="CSV (first column as text)", 
                      variable=self.format_var, value="csv").pack(anchor='w', padx=10, pady=2)
        
        # Buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="✅ Load Dataset", command=self.on_ok,
                 bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                 padx=20).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="❌ Cancel", command=self.on_cancel,
                 bg='#F44336', fg='white', font=('Arial', 10, 'bold'),
                 padx=20).pack(side=tk.LEFT, padx=5)
    
    def on_ok(self):
        """Handle OK button."""
        self.result = {
            'chunk_size': self.chunk_var.get(),
            'replace_dataset': self.replace_dataset_var.get(),
            'fit_vectorizer': self.fit_vectorizer_var.get(),
            'update_models': self.update_models_var.get(),
            'format': self.format_var.get(),
        }
        self.dialog.destroy()
    
    def on_cancel(self):
        """Handle Cancel button."""
        self.result = None
        self.dialog.destroy()

# ------------------------------------------------------
# Enhanced SNN Control Panel with Dataset Loading
# ------------------------------------------------------
class SNNControlPanel:
    """Enhanced control panel for SNN system with parameter sliders and training controls."""
    
    def __init__(self, master, text_processor, snn_model, text_generator, retrain_callback=None):
        self.master = master
        self.text_processor = text_processor
        self.snn_model = snn_model
        self.text_generator = text_generator
        self.retrain_callback = retrain_callback
        
        # Initialize variables from current model parameters
        self.variables = {
            # Core architecture parameters
            'num_neurons': tk.IntVar(value=getattr(text_processor, 'num_neurons', 128)),
            'chunk_size': tk.IntVar(value=getattr(snn_model, 'chunk_size', 32)),
            'vocab_limit': tk.IntVar(value=getattr(text_processor, 'vocab_limit', 5000)),
            'max_features': tk.IntVar(value=getattr(text_processor.vectorizer, 'max_features', 1000) if hasattr(text_processor, 'vectorizer') else 1000),
            
            # Neural network parameters
            'activation_scale1': tk.DoubleVar(value=snn_model.activation_scale1.item() if hasattr(snn_model, 'activation_scale1') else 1.0),
            'activation_scale2': tk.DoubleVar(value=snn_model.activation_scale2.item() if hasattr(snn_model, 'activation_scale2') else 1.0),
            'global_adaptation': tk.DoubleVar(value=snn_model.global_adaptation.item() if hasattr(snn_model, 'global_adaptation') else 0.5),
            
            # Text generator parameters
            'selection_sigmoid_scale': tk.DoubleVar(value=text_generator.selection_sigmoid_scale.item() if hasattr(text_generator, 'selection_sigmoid_scale') else 1.0),
            'verification_scale': tk.DoubleVar(value=text_generator.verification_scale.item() if hasattr(text_generator, 'verification_scale') else 0.8),
            'context_weight': tk.DoubleVar(value=text_generator.context_weight.item() if hasattr(text_generator, 'context_weight') else 0.3),
            'v_thresh': tk.DoubleVar(value=text_generator.v_thresh.item() if hasattr(text_generator, 'v_thresh') else 1.0),
            'quality_threshold': tk.DoubleVar(value=text_generator.quality_threshold.item() if hasattr(text_generator, 'quality_threshold') else 0.6),
            
            # Text processor parameters
            'question_weight': tk.DoubleVar(value=text_processor.question_weight.item() if hasattr(text_processor, 'question_weight') else 0.3),
            'geometric_sigmoid_scale': tk.DoubleVar(value=text_processor.geometric_sigmoid_scale.item() if hasattr(text_processor, 'geometric_sigmoid_scale') else 1.2),
            'tfidf_sigmoid_scale': tk.DoubleVar(value=text_processor.tfidf_sigmoid_scale.item() if hasattr(text_processor, 'tfidf_sigmoid_scale') else 1.0),
            
            # Training parameters
            'learning_rate': tk.DoubleVar(value=0.001),
            'epochs': tk.IntVar(value=5),
            'batch_size': tk.IntVar(value=32),
            'weight_decay': tk.DoubleVar(value=0.01),
        }
        
        self.create_widgets()
        self.update_models()  # Apply initial values

    def create_widgets(self):
        """Create the main GUI interface."""
        # Configure main window
        self.master.title("🧠 SNN Control Panel")
        self.master.geometry("900x1000")
        self.master.configure(bg='#f0f0f0')
        
        # Create main frame with scrollbar
        canvas = tk.Canvas(self.master, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(self.master, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_label = tk.Label(scrollable_frame, text="🧠 SNN & Text Generator Control Panel", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#333')
        title_label.grid(row=0, column=0, columnspan=3, pady=15, sticky='ew')
        
        row = 1
        
        # Group parameters by category
        parameter_groups = {
            "🏗️ Architecture Parameters": {
                'num_neurons': (1, 1000, 1),
                'chunk_size': (1, 100, 1),
                'vocab_limit': (1000, 100000, 1000),
                'max_features': (100, 5000, 100),
            },
            "⚡ Neural Network Parameters": {
                'activation_scale1': (0.1, 5.0, 0.1),
                'activation_scale2': (0.1, 5.0, 0.1),
                'global_adaptation': (0.0, 2.0, 0.1),
            },
            "📝 Text Generator Parameters": {
                'selection_sigmoid_scale': (0.1, 5.0, 0.1),
                'verification_scale': (0.1, 2.0, 0.1),
                'context_weight': (0.0, 1.0, 0.05),
                'v_thresh': (0.1, 5.0, 0.1),
                'quality_threshold': (0.0, 1.0, 0.05),
            },
            "🔤 Text Processor Parameters": {
                'question_weight': (0.0, 1.0, 0.05),
                'geometric_sigmoid_scale': (0.1, 5.0, 0.1),
                'tfidf_sigmoid_scale': (0.1, 5.0, 0.1),
            },
            "🎓 Training Parameters": {
                'learning_rate': (0.0001, 0.1, 0.0001),
                'epochs': (1, 20, 1),
                'batch_size': (8, 128, 8),
                'weight_decay': (0.0, 0.1, 0.001),
            }
        }
        
        # Create parameter groups
        for group_name, params in parameter_groups.items():
            # Group header
            group_label = tk.Label(scrollable_frame, text=group_name, 
                                 font=('Arial', 12, 'bold'), bg='#e0e0e0', fg='#333')
            group_label.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(10, 5), padx=5)
            row += 1
            
            # Parameters in group
            for param_name, (min_val, max_val, resolution) in params.items():
                if param_name in self.variables:
                    var = self.variables[param_name]
                    
                    # Parameter label
                    label_text = param_name.replace('_', ' ').title()
                    label = tk.Label(scrollable_frame, text=f"{label_text}:", 
                                   font=('Arial', 10), bg='#f0f0f0')
                    label.grid(row=row, column=0, sticky='e', padx=(10, 5), pady=2)
                    
                    # Slider
                    scale = tk.Scale(scrollable_frame, variable=var, 
                                   from_=min_val, to=max_val, resolution=resolution,
                                   orient='horizontal', length=300, 
                                   command=lambda val, name=param_name: self.on_parameter_change(name))
                    
                    scale.grid(row=row, column=1, sticky='w', padx=5, pady=2)
                    
                    # Value display
                    value_label = tk.Label(scrollable_frame, text=f"{var.get():.3f}", 
                                         font=('Arial', 9), bg='#f0f0f0', width=8)
                    value_label.grid(row=row, column=2, padx=(5, 10), pady=2)
                    
                    # Store reference for updates
                    setattr(self, f"{param_name}_label", value_label)
                    
                    row += 1
        
        # Control buttons frame
        button_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        # Add all buttons
        self.update_button = tk.Button(button_frame, text="🔄 Update Models", 
                                     command=self.update_models, 
                                     bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                     padx=20, pady=5)
        self.update_button.pack(side=tk.LEFT, padx=5)
        
        self.retrain_button = tk.Button(button_frame, text="🎓 Retrain", 
                                      command=self.on_retrain, 
                                      bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                                      padx=20, pady=5)
        self.retrain_button.pack(side=tk.LEFT, padx=5)
        
        # NEW: Dataset loading button
        self.load_dataset_button = tk.Button(button_frame, text="📂 Load Dataset File", 
                                           command=self.on_load_dataset_file, 
                                           bg='#673AB7', fg='white', font=('Arial', 10, 'bold'),
                                           padx=20, pady=5)
        self.load_dataset_button.pack(side=tk.LEFT, padx=5)
        
        self.save_config_button = tk.Button(button_frame, text="💾 Save Config", 
                                          command=self.save_config, 
                                          bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
                                          padx=20, pady=5)
        self.save_config_button.pack(side=tk.LEFT, padx=5)
        
        self.load_config_button = tk.Button(button_frame, text="📁 Load Config", 
                                          command=self.load_config, 
                                          bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'),
                                          padx=20, pady=5)
        self.load_config_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(button_frame, text="🔄 Reset to Default", 
                                    command=self.reset_to_default, 
                                    bg='#F44336', fg='white', font=('Arial', 10, 'bold'),
                                    padx=20, pady=5)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        row += 1
        
        # Add test panel
        row = self.create_test_panel(scrollable_frame, row)
        
        # Status label
        self.status_label = tk.Label(scrollable_frame, text="✅ Ready", 
                                   font=('Arial', 10), bg='#f0f0f0', fg='#2E7D32')
        self.status_label.grid(row=row+1, column=0, columnspan=3, pady=10)
        
        # Configure column weights
        scrollable_frame.columnconfigure(1, weight=1)

    def create_test_panel(self, parent, start_row):
        """Create the test panel for the GUI."""
        row = start_row + 1

        # Input row
        tk.Label(parent, text="Question:", font=('Arial', 10),
                 bg='#f0f0f0').grid(row=row, column=0, sticky='e', padx=(10, 5), pady=2)
        self.test_input = tk.Entry(parent, width=70)
        self.test_input.grid(row=row, column=1, sticky='we', padx=5, pady=2)
        self.test_input.bind("<Return>", lambda e: self.on_run_test())
        
        # Print to console toggle
        self.print_to_console_var = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="Print to console (CMD)",
                       variable=self.print_to_console_var, bg='#f0f0f0')\
          .grid(row=row, column=2, sticky='w', padx=(5, 10), pady=2)
        row += 1

        # Buttons
        btn_frame = tk.Frame(parent, bg='#f0f0f0')
        btn_frame.grid(row=row, column=0, columnspan=3, sticky='w', padx=(10,5), pady=(2,8))
        
        self.run_test_btn = tk.Button(btn_frame, text="▶ Run Test",
                                      command=self.on_run_test,
                                      bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                                      padx=12, pady=3)
        self.run_test_btn.pack(side=tk.LEFT, padx=(0,6))

        self.clear_test_btn = tk.Button(btn_frame, text="🧹 Clear",
                                        command=self.on_clear_test,
                                        bg='#9E9E9E', fg='white', font=('Arial', 10, 'bold'),
                                        padx=12, pady=3)
        self.clear_test_btn.pack(side=tk.LEFT, padx=(0,6))

        self.copy_test_btn = tk.Button(btn_frame, text="📋 Copy",
                                       command=self.on_copy_test,
                                       bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                       padx=12, pady=3)
        self.copy_test_btn.pack(side=tk.LEFT, padx=(0,6))
        row += 1

        # Output label
        tk.Label(parent, text="Response:", font=('Arial', 10),
                 bg='#f0f0f0').grid(row=row, column=0, sticky='ne', padx=(10, 5), pady=2)

        # Scrolled output
        self.test_output = scrolledtext.ScrolledText(parent, width=90, height=12, wrap='word')
        self.test_output.grid(row=row, column=1, columnspan=2, sticky='we', padx=5, pady=2)
        row += 1

        # Quality label
        self.test_quality = tk.Label(parent, text="Quality: N/A",
                                     font=('Arial', 10), bg='#f0f0f0', fg='#333')
        self.test_quality.grid(row=row, column=0, columnspan=3, sticky='w', padx=(10,5), pady=(2,10))
        row += 1

        return row

    # NEW: Dataset loading methods
    def on_load_dataset_file(self):
        """Handle loading dataset from file via GUI."""
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"), 
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if not filename:
            return
            
        try:
            self.status_label.config(text="📂 Loading dataset from file...", fg='#FF9800')
            self.master.update()
            
            # Show loading dialog with options
            dialog = DatasetLoadDialog(self.master, filename)
            self.master.wait_window(dialog.dialog)
            
            if not dialog.result:
                self.status_label.config(text="❌ Dataset loading cancelled", fg='#D32F2F')
                return
                
            load_options = dialog.result
            
            # Start loading in separate thread to avoid freezing GUI
            self.start_dataset_load_thread(filename, load_options)
            
        except Exception as e:
            self.status_label.config(text=f"❌ Load failed: {str(e)}", fg='#D32F2F')
            messagebox.showerror("Error", f"Failed to load dataset file: {str(e)}")

    def start_dataset_load_thread(self, filename, options):
        """Start dataset loading in separate thread."""
        def load_worker():
            try:
                self.load_dataset_button.config(state='disabled')
                
                # Clear existing data if requested
                if options.get('replace_dataset', False):
                    self.text_processor.word_to_idx = {}
                    self.text_processor.bigram_counts = Counter()
                    self.text_processor.trigram_counts = Counter()
                    self.text_processor.is_vectorizer_fitted = False
                
                # Call the text processor's loading method with format handling
                if options.get('format') == 'csv':
                    words_processed = self.load_csv_dataset(filename, options)
                else:
                    words_processed = self.text_processor.load_and_process_text_streaming(
                        file_path=filename,
                        chunk_size=options.get('chunk_size', 1000)
                    )
                
                # Update vocab size display if needed
                vocab_size = len(self.text_processor.word_to_idx)
                
                # Update models if requested
                if options.get('update_models', True):
                    self.update_models()
                
                # Update GUI on main thread
                self.master.after(0, lambda: self._update_dataset_load_ui(
                    len(words_processed), vocab_size, filename
                ))
                
            except Exception as e:
                error_msg = f"Dataset loading failed: {str(e)}"
                self.master.after(0, lambda: self._update_dataset_load_ui(
                    None, None, filename, error_msg
                ))
            
            finally:
                self.master.after(0, lambda: self.load_dataset_button.config(state='normal'))
        
        threading.Thread(target=load_worker, daemon=True).start()

    def load_csv_dataset(self, filename, options):
        """Load CSV dataset with first column as text."""
        words_processed = []
        documents = []
        current_doc = []
        word_count = 0
        
        with open(filename, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader):
                if not row or len(row) == 0:
                    continue
                    
                # Skip header row if it looks like headers
                if row_num == 0 and any(header in row[0].lower() for header in ['text', 'content', 'data', 'sample']):
                    continue
                    
                text = row[0]  # Use first column as text
                words = text.lower().split()
                word_history = []
                
                for word in words:
                    if len(self.text_processor.word_to_idx) < self.text_processor.vocab_limit:
                        if word not in self.text_processor.word_to_idx:
                            self.text_processor.word_to_idx[word] = len(self.text_processor.word_to_idx)
                    
                    word_history.append(word)
                    
                    # Build n-gram models
                    if len(word_history) >= 2:
                        self.text_processor.bigram_counts[(word_history[-2], word_history[-1])] += 1
                    if len(word_history) >= 3:
                        self.text_processor.trigram_counts[(word_history[-3], word_history[-2], word_history[-1])] += 1
                        
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
                
                if KB_LEN > 0 and word_count >= KB_LEN:
                    break
        
        if current_doc:
            documents.append(' '.join(current_doc))
        
        # Fit vectorizer if requested and not already fitted
        if options.get('fit_vectorizer', True) and documents and not self.text_processor.is_vectorizer_fitted:
            self.text_processor.fit_vectorizer(documents)
        
        return words_processed

    def _update_dataset_load_ui(self, words_count, vocab_size, filename, error=None):
        """Update UI after dataset loading completes."""
        if error:
            self.status_label.config(text=f"❌ {error}", fg='#D32F2F')
            messagebox.showerror("Dataset Load Error", error)
        else:
            success_msg = f"✅ Dataset loaded: {words_count} words, vocab: {vocab_size}"
            self.status_label.config(text=success_msg, fg='#2E7D32')
            
            messagebox.showinfo("Dataset Loaded", 
                              f"Successfully loaded dataset from:\n{Path(filename).name}\n\n"
                              f"Words processed: {words_count:,}\n"
                              f"Vocabulary size: {vocab_size:,}")

    def on_run_test(self):
        question = self.test_input.get().strip()
        if not question:
            messagebox.showinfo("Test Model", "Please enter a question to test.")
            return
        self.run_test_btn.config(state='disabled')
        self.status_label.config(text="🧪 Running test...", fg='#FF9800')
        threading.Thread(target=self._run_test_worker, args=(question,), daemon=True).start()

    def _run_test_worker(self, question: str):
        start_ts = time.time()
        try:
            words = question.lower().split()
            features = self.text_processor.words_to_neural_features(words)
            spike_outputs = self.snn_model.forward(features)
            response, quality = self.text_generator.generate_verified_response(
                spike_outputs, seed_words=words, length=200, max_attempts=3
            )

            elapsed = time.time() - start_ts
            # Update UI on main thread
            self.master.after(0, lambda: self._update_test_ui(response, quality, elapsed))

            # Optional console (CMD) logging
            if self.print_to_console_var.get():
                print("\n===== SNN Test =====")
                print(f"Question: {question}")
                print(f"Response: {response}")
                print(f"Quality: {quality:.3f} | Time: {elapsed:.3f}s")
                print("====================\n")

        except Exception as e:
            err = f"Error during test: {e}"
            self.master.after(0, lambda: self._update_test_ui(err, None, None, is_error=True))

    def _update_test_ui(self, text: str, quality: float | None, elapsed: float | None,
                        is_error: bool=False):
        self.test_output.delete('1.0', 'end')
        self.test_output.insert('end', text if text else "")
        if is_error:
            self.test_quality.config(text="Quality: N/A")
            self.status_label.config(text="❌ Test failed", fg='#D32F2F')
        else:
            qtxt = f"{quality:.3f}" if quality is not None else "N/A"
            ttxt = f" | Time: {elapsed:.3f}s" if elapsed is not None else ""
            self.test_quality.config(text=f"Quality: {qtxt}{ttxt}")
            self.status_label.config(text="✅ Test completed", fg='#2E7D32')
        self.run_test_btn.config(state='normal')

    def on_clear_test(self):
        self.test_output.delete('1.0', 'end')
        self.test_quality.config(text="Quality: N/A")

    def on_copy_test(self):
        try:
            txt = self.test_output.get('1.0', 'end').strip()
            self.master.clipboard_clear()
            self.master.clipboard_append(txt)
            self.status_label.config(text="📋 Response copied to clipboard", fg='#2E7D32')
        except Exception as e:
            self.status_label.config(text=f"❌ Copy failed: {e}", fg='#D32F2F')

    def on_parameter_change(self, param_name):
        """Handle parameter changes and update displays."""
        try:
            var = self.variables[param_name]
            value = var.get()
            
            # Update value display label
            label = getattr(self, f"{param_name}_label", None)
            if label:
                if isinstance(var, tk.IntVar):
                    label.config(text=f"{value}")
                else:
                    label.config(text=f"{value:.3f}")
            
        except Exception as e:
            print(f"Error updating parameter {param_name}: {e}")
    
    def update_models(self):
        """Update model parameters with current slider values."""
        try:
            self.status_label.config(text="🔄 Updating models...", fg='#FF9800')
            self.master.update()
            
            # Update SNN model parameters
            if hasattr(self.snn_model, 'activation_scale1'):
                self.snn_model.activation_scale1.data.fill_(self.variables['activation_scale1'].get())
            if hasattr(self.snn_model, 'activation_scale2'):
                self.snn_model.activation_scale2.data.fill_(self.variables['activation_scale2'].get())
            if hasattr(self.snn_model, 'global_adaptation'):
                self.snn_model.global_adaptation.data.fill_(self.variables['global_adaptation'].get())
            
            # Update text generator parameters
            if hasattr(self.text_generator, 'selection_sigmoid_scale'):
                self.text_generator.selection_sigmoid_scale.data.fill_(self.variables['selection_sigmoid_scale'].get())
            if hasattr(self.text_generator, 'verification_scale'):
                self.text_generator.verification_scale.data.fill_(self.variables['verification_scale'].get())
            if hasattr(self.text_generator, 'context_weight'):
                self.text_generator.context_weight.data.fill_(self.variables['context_weight'].get())
            if hasattr(self.text_generator, 'v_thresh'):
                self.text_generator.v_thresh.data.fill_(self.variables['v_thresh'].get())
            if hasattr(self.text_generator, 'quality_threshold'):
                self.text_generator.quality_threshold.data.fill_(self.variables['quality_threshold'].get())
            
            # Update text processor parameters
            if hasattr(self.text_processor, 'question_weight'):
                self.text_processor.question_weight.data.fill_(self.variables['question_weight'].get())
            if hasattr(self.text_processor, 'geometric_sigmoid_scale'):
                self.text_processor.geometric_sigmoid_scale.data.fill_(self.variables['geometric_sigmoid_scale'].get())
            if hasattr(self.text_processor, 'tfidf_sigmoid_scale'):
                self.text_processor.tfidf_sigmoid_scale.data.fill_(self.variables['tfidf_sigmoid_scale'].get())
            
            self.status_label.config(text="✅ Models updated successfully", fg='#2E7D32')
            
        except Exception as e:
            self.status_label.config(text=f"❌ Update failed: {str(e)}", fg='#D32F2F')
            messagebox.showerror("Update Error", f"Failed to update models: {str(e)}")
    
    def on_retrain(self):
        """Handle retrain button click with dialog."""
        # Create retrain dialog
        dialog = RetrainDialog(self.master, self.get_current_config())
        self.master.wait_window(dialog.dialog)
        
        if dialog.result:
            config = dialog.result
            
            # Ask if user wants to save config before retraining
            save_config = messagebox.askyesno("Save Config", 
                                            "Would you like to save this configuration before retraining?")
            if save_config:
                self.save_config_with_data(config)
            
            # Start retraining in separate thread
            self.start_retrain_thread(config)
    
    def start_retrain_thread(self, config):
        """Start retraining in a separate thread to avoid blocking the GUI."""
        def retrain_worker():
            try:
                self.status_label.config(text="🎓 Training in progress...", fg='#FF9800')
                self.retrain_button.config(state='disabled')
                self.master.update()
                
                if self.retrain_callback:
                    self.retrain_callback(config)
                else:
                    # Default retrain logic
                    self.default_retrain(config)
                
                self.status_label.config(text="✅ Training completed successfully", fg='#2E7D32')
                
            except Exception as e:
                self.status_label.config(text=f"❌ Training failed: {str(e)}", fg='#D32F2F')
                messagebox.showerror("Training Error", f"Training failed: {str(e)}")
            
            finally:
                self.retrain_button.config(state='normal')
        
        threading.Thread(target=retrain_worker, daemon=True).start()
    
    def default_retrain(self, config):
        """Default retraining logic."""
        import time
        print("Starting retrain with config:", config)
        for i in range(config['epochs']):
            print(f"Training epoch {i+1}/{config['epochs']}")
            time.sleep(1)  # Simulate training time
        print("Training completed!")
    
    def get_current_config(self):
        """Get current configuration from sliders."""
        return {k: v.get() for k, v in self.variables.items()}
    
    def save_config(self):
        """Save current configuration to JSON file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Configuration"
        )
        
        if filename:
            self.save_config_with_data(self.get_current_config(), filename)
    
    def save_config_with_data(self, config, filename=None):
        """Save configuration data to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"snn_config_{timestamp}.json"
        
        try:
            config_data = {
                "config": config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_info": {
                    "text_processor_class": self.text_processor.__class__.__name__,
                    "snn_model_class": self.snn_model.__class__.__name__,
                    "text_generator_class": self.text_generator.__class__.__name__
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            self.status_label.config(text=f"✅ Config saved to {filename}", fg='#2E7D32')
            messagebox.showinfo("Success", f"Configuration saved to {filename}")
            
        except Exception as e:
            self.status_label.config(text=f"❌ Save failed: {str(e)}", fg='#D32F2F')
            messagebox.showerror("Save Error", f"Failed to save configuration: {str(e)}")
    
    def load_config(self):
        """Load configuration from JSON file."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Configuration"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_data = json.load(f)
                
                # Handle both old and new format
                if "config" in config_data:
                    config = config_data["config"]
                else:
                    config = config_data
                
                # Update sliders with loaded values
                for param_name, value in config.items():
                    if param_name in self.variables:
                        self.variables[param_name].set(value)
                
                # Update models with loaded configuration
                self.update_models()
                
                self.status_label.config(text=f"✅ Config loaded from {filename}", fg='#2E7D32')
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
                
            except Exception as e:
                self.status_label.config(text=f"❌ Load failed: {str(e)}", fg='#D32F2F')
                messagebox.showerror("Load Error", f"Failed to load configuration: {str(e)}")
    
    def reset_to_default(self):
        """Reset all parameters to default values."""
        if messagebox.askyesno("Reset Confirmation", 
                              "Are you sure you want to reset all parameters to default values?"):
            
            default_values = {
                'num_neurons': 128,
                'chunk_size': 32,
                'vocab_limit': 5000,
                'max_features': 1000,
                'activation_scale1': 1.0,
                'activation_scale2': 1.0,
                'global_adaptation': 0.5,
                'selection_sigmoid_scale': 1.0,
                'verification_scale': 0.8,
                'context_weight': 0.3,
                'v_thresh': 1.0,
                'quality_threshold': 0.6,
                'question_weight': 0.3,
                'geometric_sigmoid_scale': 1.2,
                'tfidf_sigmoid_scale': 1.0,
                'learning_rate': 0.001,
                'epochs': 5,
                'batch_size': 32,
                'weight_decay': 0.01,
            }
            
            for param_name, default_value in default_values.items():
                if param_name in self.variables:
                    self.variables[param_name].set(default_value)
            
            self.update_models()
            self.status_label.config(text="✅ Reset to default values", fg='#2E7D32')

# ------------------------------------------------------
# Enhanced Text Generator with Question Verification - ENHANCED
# ------------------------------------------------------
class TrainableStreamingTextGenerator(nn.Module):
    def __init__(self, text_processor, hidden_dim=128, max_transitions_per_word=50):
        super().__init__()
        self.text_processor = text_processor
        self.max_transitions = max_transitions_per_word
        self.fallback_words = ["the", "and", "to", "of", "a", "in", "is", "it", "you", "that"]
        
        # Quality patterns for enhanced text generation
        self.quality_patterns = {
            'complete_answers': ['because', 'therefore', 'since', 'due', 'result', 'explanation', 'reason'],
            'explanatory_terms': ['explain', 'describe', 'define', 'meaning', 'concept', 'understand', 'clarify']
        }
        
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
        
    def verify_answer_quality(self, generated_text, original_question=None):
        """Verify the quality of generated answers."""
        text_lower = generated_text.lower()
        quality_score = 0.0
        
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
            v_mem = torch.tensor(0.0, device=next(self.parameters()).device)
            i_syn = torch.tensor(0.0, device=next(self.parameters()).device)
        
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
# Math Processor
# ------------------------------------------------------
class MathProcessor(nn.Module):
    """Mathematical processor implementing construction principles."""
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.compass_radius_scale = nn.Parameter(torch.tensor(1.0))
    
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
# Retrain Dialog
# ------------------------------------------------------
class RetrainDialog:
    """Dialog for retrain configuration."""
    
    def __init__(self, parent, current_config):
        self.result = None
        self.current_config = current_config
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("🎓 Retrain Configuration")
        self.dialog.geometry("400x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create dialog widgets."""
        # Title
        title_label = tk.Label(self.dialog, text="🎓 Configure Retraining", 
                              font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # Configuration frame
        config_frame = tk.Frame(self.dialog)
        config_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Training parameters
        tk.Label(config_frame, text="Training Parameters:", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # Epochs
        epochs_frame = tk.Frame(config_frame)
        epochs_frame.pack(fill='x', pady=2)
        tk.Label(epochs_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=self.current_config.get('epochs', 5))
        epochs_spinbox = tk.Spinbox(epochs_frame, from_=1, to=50, textvariable=self.epochs_var)
        epochs_spinbox.pack(side=tk.RIGHT)
        
        # Learning rate
        lr_frame = tk.Frame(config_frame)
        lr_frame.pack(fill='x', pady=2)
        tk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_var = tk.DoubleVar(value=self.current_config.get('learning_rate', 0.001))
        lr_spinbox = tk.Spinbox(lr_frame, from_=0.0001, to=0.1, increment=0.0001, 
                               format="%.4f", textvariable=self.lr_var)
        lr_spinbox.pack(side=tk.RIGHT)
        
        # Batch size
        batch_frame = tk.Frame(config_frame)
        batch_frame.pack(fill='x', pady=2)
        tk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_var = tk.IntVar(value=self.current_config.get('batch_size', 32))
        batch_spinbox = tk.Spinbox(batch_frame, from_=1, to=128, textvariable=self.batch_var)
        batch_spinbox.pack(side=tk.RIGHT)
        
        # Options frame
        options_frame = tk.LabelFrame(config_frame, text="Options")
        options_frame.pack(fill='x', pady=10)
        
        self.save_checkpoints_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Save checkpoints", 
                      variable=self.save_checkpoints_var).pack(anchor='w')
        
        self.use_validation_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Use validation", 
                      variable=self.use_validation_var).pack(anchor='w')
        
        self.early_stopping_var = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Early stopping", 
                      variable=self.early_stopping_var).pack(anchor='w')
        
        # Dataset options
        dataset_frame = tk.LabelFrame(config_frame, text="Dataset")
        dataset_frame.pack(fill='x', pady=10)
        
        self.dataset_var = tk.StringVar(value="multi")
        tk.Radiobutton(dataset_frame, text="Multi-dataset (Recommended)", 
                      variable=self.dataset_var, value="multi").pack(anchor='w')
        tk.Radiobutton(dataset_frame, text="Local file", 
                      variable=self.dataset_var, value="local").pack(anchor='w')
        tk.Radiobutton(dataset_frame, text="Current data", 
                      variable=self.dataset_var, value="current").pack(anchor='w')
        
        # Buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="✅ Start Training", command=self.on_ok,
                 bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                 padx=20).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="❌ Cancel", command=self.on_cancel,
                 bg='#F44336', fg='white', font=('Arial', 10, 'bold'),
                 padx=20).pack(side=tk.LEFT, padx=5)
    
    def on_ok(self):
        """Handle OK button."""
        config = self.current_config.copy()
        config.update({
            'epochs': self.epochs_var.get(),
            'learning_rate': self.lr_var.get(),
            'batch_size': self.batch_var.get(),
            'save_checkpoints': self.save_checkpoints_var.get(),
            'use_validation': self.use_validation_var.get(),
            'early_stopping': self.early_stopping_var.get(),
            'dataset_type': self.dataset_var.get(),
        })
        
        self.result = config
        self.dialog.destroy()
    
    def on_cancel(self):
        """Handle Cancel button."""
        self.result = None
        self.dialog.destroy()

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
        with open(Filename, 'r', encoding='utf-8') as f:
            text = f.read()
        vocab = text.split()
        words_processed = []
        documents = []
        current_doc = []
        word_count = 0
        # Initialize vocabulary with geometric terms
        
        for word in vocab:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        print(f"📚 Processing {len(quality_samples)} quality samples...")
        quality_samples = text.split(".")
        for sample in quality_samples:
            if KB_LEN > 0 and word_count >= KB_LEN:
                break
                
            text = sample
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
        
    def load_and_process_text_streaming(self, file_path="test.txt", chunk_size=1000, dataset_name=None, split=None, file_format="auto"):
        """Enhanced version with CSV support."""
        
        if file_format == "auto":
            # Auto-detect format based on extension
            if file_path.lower().endswith('.csv'):
                file_format = "csv"
            else:
                file_format = "txt"
        
        word_count = 0
        documents = []
        current_doc = []
        words_processed = []
        
        # Initialize vocabulary with geometric terms
        vocab = list(self.geometric_terms.keys()) + list(self.question_patterns.keys())
        for word in vocab:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        try:
            if file_format == "csv":
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:
                            continue
                        text = row[0]  # Use first column as text
                        words = text.lower().split()
                        # Process words similar to existing logic...
                        # [rest of word processing logic here]
            else:
                # Existing text file processing logic
                with open(file_path, 'r', encoding='utf-8') as f:
                    # [existing logic here]
                    pass
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error processing dataset file: {str(e)}")
        
        # Fit vectorizer if requested
        if documents and not self.is_vectorizer_fitted:
            self.fit_vectorizer(documents)
        
        print(f"📚 Loaded dataset: {word_count} words, vocab: {len(self.word_to_idx)}")
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

# Define retrain callback
def retrain_callback(config):
    """Fixed training with training pairs from loaded dataset."""
    try:
        print(f"🎓 Starting training with config: {config}")
        
        # Use parameter IDs to avoid tensor comparison ambiguity
        param_ids = set()
        all_params = []
        
        for model in [snn_model, text_generator, text_processor]:
            for param in model.parameters():
                param_id = id(param)
                if param_id not in param_ids:
                    all_params.append(param)
                    param_ids.add(param_id)
        
        optimizer = torch.optim.Adam(
            all_params,
            lr=config['learning_rate'], 
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        criterion = nn.MSELoss()
        
        # ✅ EXTRACT TRAINING PAIRS FROM LOADED DATASET
        print("📚 Extracting training pairs from loaded dataset...")
        training_pairs = []
        
        # Method 1: From existing bigram counts (if data already loaded)
        if hasattr(text_processor, 'bigram_counts') and text_processor.bigram_counts:
            print(f"📈 Found {len(text_processor.bigram_counts)} bigram patterns")
            for (w1, w2), count in text_processor.bigram_counts.items():
                if count > 1:  # Only frequent pairs
                    # Add multiple copies for frequent pairs (weighted sampling)
                    for _ in range(min(count, 5)):  # Max 5 copies per pair
                        training_pairs.append((w1, w2))
            print(f"📊 Extracted {len(training_pairs)} training pairs from bigrams")
        
        # Method 2: Load fresh data from file and extract pairs
        if len(training_pairs) < 50:  # If not enough pairs, load from file
            training_pairs.extend(load_training_pairs_from_file())
        
        # Method 3: Extract from loaded datasets (if using multi-dataset)
        if len(training_pairs) < 50 and hasattr(text_processor, 'datasets'):
            training_pairs.extend(extract_pairs_from_datasets(text_processor))
        
        if not training_pairs:
            print("❌ No training pairs available from dataset")
            return False
        
        print(f"✅ Total training pairs: {len(training_pairs)}")
        
        # Continue with training loop...
        batch_size = min(config['batch_size'], len(training_pairs))
        
        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            batch_count = 0
            
            import random
            random.shuffle(training_pairs)
            
            for i in range(0, len(training_pairs), batch_size):
                batch = training_pairs[i:i + batch_size]
                
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, requires_grad=True, device=device)
                valid_pairs = 0
                
                for w1, w2 in batch:
                    try:
                        # Forward pass
                        input_features = text_processor.words_to_neural_features([w1])
                        if input_features.numel() == 0:
                            continue
                            
                        snn_output = snn_model.forward(input_features)
                        if snn_output.numel() == 0:
                            continue
                            
                        # Get prediction
                        if snn_output.dim() > 1:
                            generator_input = snn_output.mean(dim=0).unsqueeze(0)
                        else:
                            generator_input = snn_output.unsqueeze(0)
                            
                        prediction = text_generator.forward(generator_input)
                        target_features = text_processor.words_to_neural_features([w2])
                        
                        # Ensure compatible shapes
                        min_size = min(prediction.shape[-1], target_features.shape[-1])
                        prediction_resized = prediction[..., :min_size]
                        target_resized = target_features[..., :min_size]
                        
                        pair_loss = criterion(prediction_resized, target_resized)
                        batch_loss = batch_loss + pair_loss
                        valid_pairs += 1
                        
                    except Exception as e:
                        continue
                
                if valid_pairs > 0 and batch_loss.requires_grad:
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step()
                    epoch_loss += batch_loss.item()
                    batch_count += 1
                
                if batch_count % 5 == 0 and batch_count > 0:
                    current_loss = batch_loss.item() if hasattr(batch_loss, 'item') else 0
                    print(f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_count}, Loss: {current_loss:.4f}")
            
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"✅ Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        print("🎉 Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_training_pairs_from_file():
    """Load training pairs directly from file."""
    training_pairs = []
    
    try:
        # Try to load from your dataset file
        filename = self.filename  # "xaa" or your dataset file
        if Path(filename).exists():
            print(f"📁 Loading training pairs from {filename}")
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            print(f"⚠️ File {filename} not found, using synthetic data")
            text = """
            What is geometry construction compass circle intersection mathematical proof theorem euclidean point line angle radius center.
            How to construct geometric shapes using compass only constructions without straightedge.
            Explain the Mohr-Mascheroni theorem and its applications in mathematical proofs.
            """
        
        # Extract word pairs from text
        words = text.lower().split()
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            # Filter out very short words and punctuation
            if len(w1) > 2 and len(w2) > 2 and w1.isalpha() and w2.isalpha():
                training_pairs.append((w1, w2))
        
        print(f"📊 Extracted {len(training_pairs)} pairs from file")
        
    except Exception as e:
        print(f"⚠️ Error loading from file: {e}")
    
    return training_pairs

def extract_pairs_from_datasets(text_processor):
    """Extract training pairs from loaded datasets."""
    training_pairs = []
    
    try:
        if hasattr(text_processor, 'datasets') and text_processor.datasets:
            for dataset_info in text_processor.datasets[:2]:  # Use first 2 datasets
                print(f"📊 Extracting pairs from {dataset_info['name']}")
                
                if isinstance(dataset_info['data'], list):
                    # Synthetic dataset
                    for sample in dataset_info['data'][:50]:  # Limit samples
                        text = sample.get('text', '')
                        words = text.lower().split()
                        for i in range(len(words) - 1):
                            if len(words[i]) > 2 and len(words[i+1]) > 2:
                                training_pairs.append((words[i], words[i+1]))
                else:
                    # HuggingFace dataset
                    text_field = dataset_info['text_field']
                    for i, sample in enumerate(dataset_info['data']):
                        if i >= 100:  # Limit to 100 samples
                            break
                        try:
                            text = str(sample[text_field])
                            words = text.lower().split()
                            for j in range(len(words) - 1):
                                if len(words[j]) > 2 and len(words[j+1]) > 2:
                                    training_pairs.append((words[j], words[j+1]))
                        except:
                            continue
                
                print(f"📈 Got {len(training_pairs)} pairs from {dataset_info['name']}")
                
                if len(training_pairs) > 1000:  # Limit total pairs
                    break
                    
    except Exception as e:
        print(f"⚠️ Error extracting from datasets: {e}")
    
    return training_pairs


def prepare_training_data(text_processor, batch_size=32):
    """Prepare training data batches."""
    training_pairs = []
    
    # Extract word sequences from bigram/trigram data
    for (w1, w2), count in list(text_processor.bigram_counts.items())[:1000]:
        if count > 2:  # Only use frequent pairs
            training_pairs.append(([w1], [w2]))
    
    # Create batches
    for i in range(0, len(training_pairs), batch_size):
        batch = training_pairs[i:i + batch_size]
        if batch:
            yield batch[0]  # Simplified - return first pair for now
            
def retrain_callback(config):
    """Fixed training with training pairs from loaded dataset."""
    try:
        print(f"🎓 Starting training with config: {config}")
        
        # Use parameter IDs to avoid tensor comparison ambiguity
        param_ids = set()
        all_params = []
        
        for model in [snn_model, text_generator, text_processor]:
            for param in model.parameters():
                param_id = id(param)
                if param_id not in param_ids:
                    all_params.append(param)
                    param_ids.add(param_id)
        
        optimizer = torch.optim.Adam(
            all_params,
            lr=config['learning_rate'], 
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        criterion = nn.MSELoss()
        
        # ✅ EXTRACT TRAINING PAIRS FROM LOADED DATASET
        print("📚 Extracting training pairs from loaded dataset...")
        training_pairs = []
        
        # Method 1: From existing bigram counts (if data already loaded)
        if hasattr(text_processor, 'bigram_counts') and text_processor.bigram_counts:
            print(f"📈 Found {len(text_processor.bigram_counts)} bigram patterns")
            for (w1, w2), count in text_processor.bigram_counts.items():
                if count > 1:  # Only frequent pairs
                    # Add multiple copies for frequent pairs (weighted sampling)
                    for _ in range(min(count, 5)):  # Max 5 copies per pair
                        training_pairs.append((w1, w2))
            print(f"📊 Extracted {len(training_pairs)} training pairs from bigrams")
        
        # Method 2: Load fresh data from file and extract pairs
        if len(training_pairs) < 50:  # If not enough pairs, load from file
            training_pairs.extend(load_training_pairs_from_file())
        
        # Method 3: Extract from loaded datasets (if using multi-dataset)
        if len(training_pairs) < 50 and hasattr(text_processor, 'datasets'):
            training_pairs.extend(extract_pairs_from_datasets(text_processor))
        
        if not training_pairs:
            print("❌ No training pairs available from dataset")
            return False
        
        print(f"✅ Total training pairs: {len(training_pairs)}")
        
        # Continue with training loop...
        batch_size = min(config['batch_size'], len(training_pairs))
        
        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            batch_count = 0
            
            import random
            random.shuffle(training_pairs)
            
            for i in range(0, len(training_pairs), batch_size):
                batch = training_pairs[i:i + batch_size]
                
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, requires_grad=True, device=device)
                valid_pairs = 0
                
                for w1, w2 in batch:
                    try:
                        # Forward pass
                        input_features = text_processor.words_to_neural_features([w1])
                        if input_features.numel() == 0:
                            continue
                            
                        snn_output = snn_model.forward(input_features)
                        if snn_output.numel() == 0:
                            continue
                            
                        # Get prediction
                        if snn_output.dim() > 1:
                            generator_input = snn_output.mean(dim=0).unsqueeze(0)
                        else:
                            generator_input = snn_output.unsqueeze(0)
                            
                        prediction = text_generator.forward(generator_input)
                        target_features = text_processor.words_to_neural_features([w2])
                        
                        # Ensure compatible shapes
                        min_size = min(prediction.shape[-1], target_features.shape[-1])
                        prediction_resized = prediction[..., :min_size]
                        target_resized = target_features[..., :min_size]
                        
                        pair_loss = criterion(prediction_resized, target_resized)
                        batch_loss = batch_loss + pair_loss
                        valid_pairs += 1
                        
                    except Exception as e:
                        continue
                
                if valid_pairs > 0 and batch_loss.requires_grad:
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step()
                    epoch_loss += batch_loss.item()
                    batch_count += 1
                
                if batch_count % 5 == 0 and batch_count > 0:
                    current_loss = batch_loss.item() if hasattr(batch_loss, 'item') else 0
                    print(f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_count}, Loss: {current_loss:.4f}")
            
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"✅ Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        print("🎉 Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_training_pairs_from_file():
    """Load training pairs directly from file."""
    training_pairs = []
    
    try:
        # Try to load from your dataset file
        filename = Filename  # "xaa" or your dataset file
        if Path(filename).exists():
            print(f"📁 Loading training pairs from {filename}")
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            print(f"⚠️ File {filename} not found, using synthetic data")
            text = """
            What is geometry construction compass circle intersection mathematical proof theorem euclidean point line angle radius center.
            How to construct geometric shapes using compass only constructions without straightedge.
            Explain the Mohr-Mascheroni theorem and its applications in mathematical proofs.
            """
        
        # Extract word pairs from text
        words = text.lower().split()
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            # Filter out very short words and punctuation
            if len(w1) > 2 and len(w2) > 2 and w1.isalpha() and w2.isalpha():
                training_pairs.append((w1, w2))
        
        print(f"📊 Extracted {len(training_pairs)} pairs from file")
        
    except Exception as e:
        print(f"⚠️ Error loading from file: {e}")
    
    return training_pairs

def extract_pairs_from_datasets(text_processor):
    """Extract training pairs from loaded datasets."""
    training_pairs = []
    
    try:
        if hasattr(text_processor, 'datasets') and text_processor.datasets:
            for dataset_info in text_processor.datasets[:2]:  # Use first 2 datasets
                print(f"📊 Extracting pairs from {dataset_info['name']}")
                
                if isinstance(dataset_info['data'], list):
                    # Synthetic dataset
                    for sample in dataset_info['data'][:50]:  # Limit samples
                        text = sample.get('text', '')
                        words = text.lower().split()
                        for i in range(len(words) - 1):
                            if len(words[i]) > 2 and len(words[i+1]) > 2:
                                training_pairs.append((words[i], words[i+1]))
                else:
                    # HuggingFace dataset
                    text_field = dataset_info['text_field']
                    for i, sample in enumerate(dataset_info['data']):
                        if i >= 100:  # Limit to 100 samples
                            break
                        try:
                            text = str(sample[text_field])
                            words = text.lower().split()
                            for j in range(len(words) - 1):
                                if len(words[j]) > 2 and len(words[j+1]) > 2:
                                    training_pairs.append((words[j], words[j+1]))
                        except:
                            continue
                
                print(f"📈 Got {len(training_pairs)} pairs from {dataset_info['name']}")
                
                if len(training_pairs) > 1000:  # Limit total pairs
                    break
                    
    except Exception as e:
        print(f"⚠️ Error extracting from datasets: {e}")
    
    return training_pairs






# Create GUI
root = tk.Tk()
control_panel = SNNControlPanel(root, text_processor, snn_model, text_generator, retrain_callback)

print("🚀 SNN Control Panel launched!")
print("📊 Adjust parameters using the sliders")
print("🔄 Click 'Update Models' to apply changes")
print("🎓 Click 'Retrain' to start training with current parameters")

root.mainloop()
