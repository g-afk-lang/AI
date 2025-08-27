import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import json
import threading
from pathlib import Path
import time
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
        self.master.geometry("800x900")
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
                    if isinstance(var, tk.IntVar):
                        scale = tk.Scale(scrollable_frame, variable=var, 
                                       from_=min_val, to=max_val, resolution=resolution,
                                       orient='horizontal', length=300, 
                                       command=lambda val, name=param_name: self.on_parameter_change(name))
                    else:
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
        
        # Buttons
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
        
        # Status label
        self.status_label = tk.Label(scrollable_frame, text="✅ Ready", 
                                   font=('Arial', 10), bg='#f0f0f0', fg='#2E7D32')
        self.status_label.grid(row=row+1, column=0, columnspan=3, pady=10)
        
        # Configure column weights
        scrollable_frame.columnconfigure(1, weight=1)
    
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
            
            # Auto-update models on parameter change (optional, can be disabled for performance)
            # self.update_models()
            
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
        # This would be replaced with actual retraining logic
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


# Enhanced main implementation with GUI
def main_implementation_with_gui():
    """Enhanced main implementation with tkinter GUI."""
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Initialize models (same as before)
    num_neurons = 128
    chunk_size = 16
    vocab_limit = 50000
    max_features = 500
    
    # Import your existing classes here
    # text_processor = EnhancedTextProcessor(num_neurons, device=device, vocab_limit=vocab_limit, max_features=max_features).to(device)
    # snn_model = TrainableStreamingSNN(num_neurons, device=device, chunk_size=chunk_size).to(device)
    # text_generator = TrainableStreamingTextGenerator(text_processor, hidden_dim=256).to(device)
    
    # For demonstration, create dummy models
    class DummyModel:
        def __init__(self):
            import torch.nn as nn
            self.activation_scale1 = nn.Parameter(torch.tensor(1.0))
            self.activation_scale2 = nn.Parameter(torch.tensor(1.0))
            self.global_adaptation = nn.Parameter(torch.tensor(0.5))
            self.chunk_size = 32
    
    class DummyTextProcessor:
        def __init__(self):
            import torch.nn as nn
            self.num_neurons = 128
            self.vocab_limit = 5000
            self.question_weight = nn.Parameter(torch.tensor(0.3))
            self.geometric_sigmoid_scale = nn.Parameter(torch.tensor(1.2))
            self.tfidf_sigmoid_scale = nn.Parameter(torch.tensor(1.0))
            
            class MockVectorizer:
                max_features = 1000
            self.vectorizer = MockVectorizer()
    
    class DummyTextGenerator:
        def __init__(self):
            import torch.nn as nn
            self.selection_sigmoid_scale = nn.Parameter(torch.tensor(1.0))
            self.verification_scale = nn.Parameter(torch.tensor(0.8))
            self.context_weight = nn.Parameter(torch.tensor(0.3))
            self.v_thresh = nn.Parameter(torch.tensor(1.0))
            self.quality_threshold = nn.Parameter(torch.tensor(0.6))
    
    # Create dummy models for demonstration
    text_processor = DummyTextProcessor()
    snn_model = DummyModel()
    text_generator = DummyTextGenerator()
    
    # Define retrain callback
    def retrain_callback(config):
        """Custom retrain function."""
        print(f"🎓 Starting retrain with config: {config}")
        # Add your actual retraining logic here
        # This would call train_snn_system with the new configuration
        return True
    
    # Create GUI
    root = tk.Tk()
    control_panel = SNNControlPanel(root, text_processor, snn_model, text_generator, retrain_callback)
    
    print("🚀 SNN Control Panel launched!")
    print("📊 Adjust parameters using the sliders")
    print("🔄 Click 'Update Models' to apply changes")
    print("🎓 Click 'Retrain' to start training with current parameters")
    
    root.mainloop()

if __name__ == "__main__":
    # For testing the GUI independently
    main_implementation_with_gui()
