import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import json
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
import numpy as np

class ModernSentimentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Advanced Sentiment Analysis Studio")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize status variable first
        self.status_var = tk.StringVar(value="Initializing...")
        
        # Configure modern styling
        self.setup_styles()
        self.create_widgets()
        self.load_models()
        
        # Center the window
        self.center_window()
    
    def setup_styles(self):
        """Configure modern ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 24, 'bold'),
                       foreground='#2c3e50',
                       background='#f0f0f0')
        
        style.configure('Subtitle.TLabel',
                       font=('Segoe UI', 11),
                       foreground='#34495e',
                       background='#f0f0f0')
        
        style.configure('Modern.TButton',
                       font=('Segoe UI', 12, 'bold'),
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       relief='flat')
        
        style.map('Modern.TButton',
                 background=[('active', '#2980b9'),
                           ('pressed', '#1f4e79'),
                           ('!active', '#3498db')])
        
        style.configure('Clear.TButton',
                       font=('Segoe UI', 10),
                       foreground='#e74c3c',
                       background='#f8f9fa',
                       borderwidth=1,
                       relief='solid')
        
        style.map('Clear.TButton',
                 background=[('active', '#fadbd8'),
                           ('pressed', '#f1948a')])
        
        style.configure('Result.TLabelFrame',
                       font=('Segoe UI', 12, 'bold'),
                       foreground='#2c3e50',
                       background='#f0f0f0',
                       borderwidth=2,
                       relief='solid')
        
        style.configure('Card.TFrame',
                       background='white',
                       borderwidth=1,
                       relief='solid')
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def load_models(self):
        """Load the required ML models and vectorizer"""
        self.status_var.set("🔄 Loading models...")
        self.root.update()
        
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Load baseline model and vectorizer
            self.status_var.set("🔄 Loading baseline model...")
            self.root.update()
            self.baseline_model = joblib.load(os.path.join(base_dir, "models", "baseline", "sentiment_baseline_model.pkl"))
            self.vectorizer = joblib.load(os.path.join(base_dir, "models", "baseline", "sentiment_baseline_vectorizer.pkl"))
            
            # Load LSTM model
            self.status_var.set("🔄 Loading LSTM model...")
            self.root.update()
            self.lstm_model = load_model(os.path.join(base_dir, "models", "lstm_model.h5"))

            # Load tokenizer
            self.status_var.set("🔄 Loading tokenizer...")
            self.root.update()
            tok_path = os.path.join(base_dir, "models", "tokenizer.json")
            if os.path.exists(tok_path):
                with open(tok_path, "r", encoding="utf-8") as f:
                    tok_obj = json.load(f)
                tok_json = tok_obj if isinstance(tok_obj, str) else json.dumps(tok_obj)
                self.tokenizer = tokenizer_from_json(tok_json)
            else:
                self.tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

            self.max_seq_len = 100
            self.ready = True
            self.status_var.set("✅ Models loaded successfully - Ready for analysis")

        except Exception as e:
            self.ready = False
            self.status_var.set("❌ Failed to load models")
            messagebox.showerror("Model Loading Error", 
                               f"Failed to load models/vectorizer.\n\nError: {str(e)}\n\n"
                               f"Please ensure all model files are in the correct location:\n"
                               f"- models/baseline/sentiment_baseline_model.pkl\n"
                               f"- models/baseline/sentiment_baseline_vectorizer.pkl\n"
                               f"- models/lstm_model.h5\n"
                               f"- models/tokenizer.json (optional)")
    
    def get_sentiment_color(self, sentiment):
        """Return color based on sentiment"""
        colors = {
            'Positive': '#27ae60',
            'Neutral': '#f39c12', 
            'Negative': '#e74c3c'
        }
        return colors.get(sentiment, '#95a5a6')
    
    def get_sentiment_emoji(self, sentiment):
        """Return emoji based on sentiment"""
        emojis = {
            'Positive': '😊',
            'Neutral': '😐',
            'Negative': '😔'
        }
        return emojis.get(sentiment, '❓')
    
    def predict_sentiment(self):
        """Get prediction for the input text with enhanced visualization"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return
        
        try:
            if not getattr(self, "ready", False):
                messagebox.showerror("Error", "Models are not loaded. Please fix paths and restart the app.")
                return
            
            # Show processing status
            self.status_var.set("🔄 Analyzing sentiment...")
            self.root.update()
            
            # Baseline prediction
            X_vec = self.vectorizer.transform([text])
            baseline_pred = self.baseline_model.predict(X_vec)[0]
            baseline_proba = self.baseline_model.predict_proba(X_vec)[0]
            
            # LSTM prediction
            X_seq = self.tokenizer.texts_to_sequences([text])
            X_pad = pad_sequences(X_seq, maxlen=self.max_seq_len, padding='post', truncating='post')
            lstm_proba = self.lstm_model.predict(X_pad, verbose=0)[0]
            lstm_pred = np.argmax(lstm_proba)
            
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            baseline_result = sentiment_map.get(baseline_pred, "Unknown")
            lstm_result = sentiment_map.get(lstm_pred, "Unknown")
            
            # Update results with rich formatting
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            
            # Add styled results
            result_content = f"""📝 INPUT TEXT
{text[:200]}{'...' if len(text) > 200 else ''}

🤖 BASELINE MODEL RESULTS
Prediction: {self.get_sentiment_emoji(baseline_result)} {baseline_result}
Confidence: {max(baseline_proba):.2%}
Probabilities:
  • Negative: {baseline_proba[0]:.2%}
  • Neutral: {baseline_proba[1]:.2%}
  • Positive: {baseline_proba[2]:.2%}

🧠 LSTM MODEL RESULTS
Prediction: {self.get_sentiment_emoji(lstm_result)} {lstm_result}
Confidence: {max(lstm_proba):.2%}
Probabilities:
  • Negative: {lstm_proba[0]:.2%}
  • Neutral: {lstm_proba[1]:.2%}
  • Positive: {lstm_proba[2]:.2%}

📊 CONSENSUS
Both models agree: {'✅ Yes' if baseline_result == lstm_result else '❌ No'}
Final Prediction: {lstm_result} {self.get_sentiment_emoji(lstm_result)}
"""
            
            self.result_text.insert(tk.END, result_content)
            
            # Color-code the final prediction
            self.result_text.tag_configure("positive", foreground="#27ae60", font=("Segoe UI", 10, "bold"))
            self.result_text.tag_configure("neutral", foreground="#f39c12", font=("Segoe UI", 10, "bold"))
            self.result_text.tag_configure("negative", foreground="#e74c3c", font=("Segoe UI", 10, "bold"))
            
            self.result_text.config(state=tk.DISABLED)
            self.status_var.set("✅ Analysis complete")
            
        except Exception as e:
            self.status_var.set("❌ Analysis failed")
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def clear_all(self):
        """Clear all input and results"""
        self.text_input.delete(1.0, tk.END)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.status_var.set("Ready for analysis")
    
    def load_sample_text(self):
        """Load sample text for testing"""
        samples = [
            "I absolutely love this product! It's amazing and exceeded all my expectations.",
            "The weather is okay today, nothing special but not bad either.",
            "I'm really disappointed with the service. It was terrible and frustrating.",
            "This new restaurant has incredible food and outstanding service!",
            "The movie was average, neither great nor terrible."
        ]
        import random
        sample = random.choice(samples)
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(1.0, sample)
    
    def create_widgets(self):
        """Create and arrange the modern GUI widgets"""
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header section
        header_frame = ttk.Frame(main_container, style='Card.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_content = tk.Frame(header_frame, bg='white')
        header_content.pack(fill=tk.X, padx=20, pady=20)
        
        title_label = ttk.Label(header_content, 
                               text="🎯 Advanced Sentiment Analysis Studio",
                               style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(header_content,
                                  text="Powered by Machine Learning & Deep Learning Models",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(pady=(5, 0))
        
        # Input section
        input_frame = ttk.Frame(main_container, style='Card.TFrame')
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        input_content = tk.Frame(input_frame, bg='white')
        input_content.pack(fill=tk.X, padx=20, pady=20)
        
        input_label = ttk.Label(input_content,
                               text="💬 Enter your text for sentiment analysis:",
                               font=('Segoe UI', 12, 'bold'),
                               background='white',
                               foreground='#2c3e50')
        input_label.pack(anchor='w', pady=(0, 10))
        
        # Text input with custom styling
        text_frame = tk.Frame(input_content, bg='white')
        text_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.text_input = scrolledtext.ScrolledText(text_frame,
            wrap=tk.WORD,
            width=70,
            height=6,
            font=('Segoe UI', 11),
            borderwidth=2,
            relief='solid',
             bg='#fafafa')
        self.text_input.pack(fill=tk.X)
        
        # Button section
        button_frame = tk.Frame(input_content, bg='white')
        button_frame.pack(fill=tk.X)
        
        analyze_btn = ttk.Button(button_frame,
                                text="🔍 Analyze Sentiment",
                                command=self.predict_sentiment,
                                style='Modern.TButton')
        analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        sample_btn = ttk.Button(button_frame,
                               text="📝 Load Sample",
                               command=self.load_sample_text)
        sample_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ttk.Button(button_frame,
                              text="🗑️ Clear All",
                              command=self.clear_all,
                              style='Clear.TButton')
        clear_btn.pack(side=tk.LEFT)
        
        # Results section
        results_frame = ttk.LabelFrame(main_container,
                                      text="📊 Analysis Results",
                                      style='Result.TLabelframe')
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        results_content = tk.Frame(results_frame, bg='white')
        results_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.result_text = scrolledtext.ScrolledText(results_content,
                                                    wrap=tk.WORD,
                                                    width=70,
                                                    height=12,
                                                    font=('Segoe UI', 10),
                                                    borderwidth=1,
                                                    relief='solid',
                                                    bg='#fafafa',
                                                    state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        status_frame = tk.Frame(main_container, bg='#ecf0f1', relief='sunken', bd=1)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        status_label = tk.Label(status_frame,
                               textvariable=self.status_var,
                               font=('Segoe UI', 9),
                               bg='#ecf0f1',
                               fg='#2c3e50')
        status_label.pack(side=tk.LEFT, padx=10, pady=5)

def main():
    root = tk.Tk()
    app = ModernSentimentAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()