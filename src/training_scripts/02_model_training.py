"""
Script 02: Model Training

Loads the preprocessed clean and noisy text data,
tokenizes it, and trains the Seq2Seq model.

This script should be run AFTER 01_data_preprocessing.py.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- !! TRAINING CONFIGURATION !! ---

# Set to True for a quick test run (small data, few epochs)
# Set to False for the full, final model training
TEST_MODE = True

# --- End of Configuration ---


# --- Configuration ---
# Define paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Input files
CLEAN_TRAIN_PATH = os.path.join(DATA_DIR, 'train_clean.txt')
NOISY_TRAIN_PATH = os.path.join(DATA_DIR, 'train_noisy.txt')
TOKENIZER_PATH = os.path.join(DATA_DIR, 'tokenizer_config.json')

# Output file
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'autocorrect_model.h5')

# Model Hyperparameters (Must match 01_data_preprocessing.py)
MAX_LEN = 20
VOCAB_SIZE = 10000
EMBEDDING_DIM = 256
GRU_UNITS = 256

# --- Set Training Parameters based on TEST_MODE ---
if TEST_MODE:
    print("---!! RUNNING IN TEST MODE !!---")
    BATCH_SIZE = 64
    EPOCHS = 3
    DATA_SLICE = 20000 # Use only 20,000 samples
else:
    print("--- RUNNING IN FULL TRAINING MODE ---")
    BATCH_SIZE = 128
    EPOCHS = 50
    DATA_SLICE = None # Use all data


# --- Helper Functions ---

def load_data(path, num_samples=None):
    """Loads a text file into a list of lines, optionally slicing it."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            if num_samples:
                return lines[:num_samples]
            return lines
    except FileNotFoundError:
        print(f"CRITICAL ERROR: File not found at {path}")
        return None

def load_tokenizer(path):
    """Loads a Keras tokenizer from a JSON config."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config_string = json.load(f) # Load the JSON string from the file
            return tokenizer_from_json(config_string)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Tokenizer not found at {path}")
        return None

def build_seq2seq_model(vocab_size, max_len, embedding_dim, gru_units):
    """
    Builds and compiles the corrected Seq2Seq (many-to-many) GRU model.
    """
    print("Building Seq2Seq (many-to-many) model...")
    model = Sequential()
    
    # Input/Embedding Layer
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        mask_zero=True  # Important for handling padding
    ))
    
    # Encoder Layer
    # Using Bidirectional GRU for better context understanding
    # return_sequences=True is essential for Seq2Seq
    model.add(Bidirectional(GRU(gru_units, return_sequences=True)))
    
    # Regularization
    model.add(Dropout(0.2))
    
    # Output Layer (Decoder)
    # Apply a Dense layer to *every* time step
    # This predicts a probability distribution over the vocab for each token
    model.add(Dense(vocab_size, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # Use this loss for integer labels
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def setup_gpu_memory_growth():
    """Sets GPU memory growth to avoid OOM errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"INFO: Enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"ERROR: Could not set memory growth: {e}")

# --- Main Execution ---

def main():
    """
    Main training pipeline.
    """
    print("Starting model training...")
    setup_gpu_memory_growth()

    # 1. Load Tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    if not tokenizer:
        return
        
    # 2. Load Data (using DATA_SLICE)
    clean_lines = load_data(CLEAN_TRAIN_PATH, DATA_SLICE)
    noisy_lines = load_data(NOISY_TRAIN_PATH, DATA_SLICE)
    
    if not clean_lines or not noisy_lines or len(clean_lines) != len(noisy_lines):
        print("ERROR: Training data is missing or mismatched.")
        return
        
    print(f"Loaded {len(clean_lines)} clean and {len(noisy_lines)} noisy lines.")

    # 3. Tokenize and Pad Data
    
    # X (input) is the noisy data
    print("Tokenizing and padding 'X' (noisy) data...")
    X_seq = tokenizer.texts_to_sequences(noisy_lines)
    X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post')
    
    # Y (target) is the clean data
    print("Tokenizing and padding 'Y' (clean) data...")
    Y_seq = tokenizer.texts_to_sequences(clean_lines)
    Y_pad = pad_sequences(Y_seq, maxlen=MAX_LEN, padding='post')
    
    # We must add an extra dimension to Y for sparse_categorical_crossentropy
    # The shape needs to be (samples, timesteps, 1)
    Y_pad = np.expand_dims(Y_pad, -1)
    
    print(f"Final data shapes: X={X_pad.shape}, Y={Y_pad.shape}")

    # 4. Build Model
    model = build_seq2seq_model(VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM, GRU_UNITS)

    # 5. Define Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', # Stop if validation loss stops improving
        patience=5,         # Wait 5 epochs
        restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH, 
        save_best_only=True, 
        monitor='val_loss',
        verbose=1
    )

    # 6. Train Model
    print("Starting model fitting...")
    history = model.fit(
        X_pad,
        Y_pad,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,  # Use 10% of training data for validation
        callbacks=[early_stopping, model_checkpoint]
    )

    print(f"\nTraining complete. Best model saved to '{MODEL_SAVE_PATH}'.")
    
if __name__ == "__main__":
    main()