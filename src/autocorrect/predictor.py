"""
Handles loading the trained AI model and tokenizer
to perform predictions (corrections).
"""

import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
import warnings

# --- Path Definitions ---
# Define paths relative to the *project root*, not this file.
# This file is at: Type-Correcter-Ai/src/autocorrect/predictor.py
# The project root is 3 levels up.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'autocorrect_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer_config.json')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

class Corrector:
    """
    A class to encapsulate the trained model and tokenizer
    for making typo corrections.
    """
    
    def __init__(self, model_path, tokenizer_path):
        """
        Initializes the Corrector by loading the model and tokenizer.
        
        Args:
            model_path (str): Path to the saved .h5 model file.
            tokenizer_path (str): Path to the saved tokenizer_config.json file.
        """
        print("INFO: Loading model and tokenizer...")
        try:
            # Load the pre-trained model
            self.model = load_model(model_path, compile=False)
            
            # Load the tokenizer
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                # The tokenizer is saved as a JSON string
                tokenizer_data = json.load(f) 
                self.tokenizer = tokenizer_from_json(tokenizer_data)
            
            # Get model's expected input length
            self.max_len = self.model.input_shape[1]
            if not self.max_len:
                # Fallback for models with dynamic input.
                # We will assume 20, which is used in 02_model_training.py
                print("WARN: Could not infer max_len from model. Assuming 20.")
                self.max_len = 20
                
            # Create reverse mapping (index -> word)
            self.reverse_word_index = {v: k for k, v in self.tokenizer.word_index.items()}
            
            print(f"INFO: Model loaded successfully. Max sequence length: {self.max_len}")

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model or tokenizer.")
            print(f"Error: {e}")
            print(f"Please ensure '{model_path}' and '{tokenizer_path}' exist.")
            self.model = None
            self.tokenizer = None
            self.max_len = 0

    def predict(self, text):
        """
        Predicts the correction for a given input text.
        
        Args:
            text (str): The noisy input text.
            
        Returns:
            str: The corrected output text.
        """
        if not self.model or not self.tokenizer:
            return "Error: Model is not loaded."
            
        # 1. Preprocess the input text
        try:
            # Add <sos> and <eos> tokens, as used in training
            clean_text = f"<sos> {text.lower()} <eos>"
            
            # Convert text to sequence
            sequence = self.tokenizer.texts_to_sequences([clean_text])
            
            # Pad the sequence
            padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post')
            
            # 2. Make prediction
            prediction = self.model.predict(padded_sequence, verbose=0)
            
            # 3. Decode the prediction
            # Output shape is (batch, max_len, vocab_size)
            # Get the index of the highest probability word at each step
            output_indices = np.argmax(prediction, axis=-1)[0]
            
            return self._indices_to_text(output_indices)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Error: Prediction failed."

    def _indices_to_text(self, indices):
        """Helper function to convert a list of indices back to a string."""
        words = []
        for idx in indices:
            if idx == 0:  # Skip padding token
                continue
            word = self.reverse_word_index.get(idx)
            if word:
                if word == '<eos>': # Stop at end-of-sequence
                    break
                if word == '<sos>': # Skip start-of-sequence
                    continue
                words.append(word)
        return ' '.join(words)

# --- Singleton Instance ---
# Create a single instance of the Corrector when the app loads.
# This is crucial for performance, as it avoids re-loading the
# model (which is slow) on every single web request.
print("INFO: Initializing global Corrector instance...")
corrector_instance = Corrector(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH)