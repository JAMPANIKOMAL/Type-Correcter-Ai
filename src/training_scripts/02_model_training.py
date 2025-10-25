#!/usr/bin/env python3
"""
Ghost Type Corrector - Model Training Pipeline
===============================================
Trains a sequence-to-sequence LSTM model for autocorrection.

This script:
1. Loads preprocessed training data
2. Builds character-level tokenizer
3. Creates encoder-decoder LSTM architecture
4. Trains the model with GPU acceleration (if available)
5. Saves the trained model and tokenizer configuration

Author: Ghost Type Corrector Team
License: MIT
"""

import tensorflow as tf
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

# =============================================================================
# GPU CONFIGURATION
# =============================================================================

# Configure GPU memory growth to avoid OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Found {len(gpus)} GPU(s) - Memory growth enabled")
        print(f"  Devices: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"⚠ GPU configuration error: {e}")
else:
    print("ℹ No GPU detected - Training will use CPU")

print()

# Additional diagnostics: TensorFlow/CUDA/GPU info and small test op
try:
    print("TensorFlow version:", tf.__version__)
    print("TF built with CUDA support:", tf.test.is_built_with_cuda())
    physical = tf.config.list_physical_devices('GPU')
    print("Physical GPUs:", physical)
    logical = tf.config.list_logical_devices('GPU') if physical else []
    print("Logical GPUs:", logical)

    if physical:
        # Run a tiny op on GPU: create a tensor and multiply to validate device
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0, 3.0])
                b = a * 2.0
                # Ensure evaluation - will raise if GPU not available for ops
                print("GPU test op result:", b.numpy())
                print("GPU test: SUCCESS - operations run on /GPU:0")
        except Exception as e:
            print("GPU test op failed:", e)
    else:
        print("GPU test: SKIPPED (no physical GPU devices)")
except Exception as e:
    print("GPU diagnostics failed:", e)


# =============================================================================
# TRAINING CONFIGURATION - MAXIMUM SETTINGS
# =============================================================================

# Data loading
NUM_SAMPLES = None  # Use full dataset for maximum accuracy
MAX_SENTENCE_LENGTH = 150  # Increased for longer context

# Model architecture - MAXIMUM SETTINGS
EMBEDDING_DIM = 256  # Increased from 128 for richer representations
LATENT_DIM = 512     # Increased from 256 for more model capacity

# Training parameters - MAXIMUM SETTINGS
EPOCHS = 20          # Increased from 10 for better convergence
BATCH_SIZE = 128     # Increased for faster training (reduce if OOM)
VALIDATION_SPLIT = 0.15  # More data for training (was 0.2)

# Special tokens
START_TOKEN = '\t'   # Start of sequence
END_TOKEN = '\n'     # End of sequence
PAD_TOKEN = ''       # Padding (index 0)


# =============================================================================
# DATA LOADING AND TOKENIZATION
# =============================================================================

def load_training_data(clean_path: Path, noisy_path: Path, num_samples: int = None):
    """
    Load preprocessed training data.
    
    Args:
        clean_path: Path to clean sentences file
        noisy_path: Path to noisy sentences file
        num_samples: Number of samples to load (None = all)
        
    Returns:
        Tuple of (clean_lines, noisy_lines)
    """
    print(f"Loading data from:")
    print(f"  Clean: {clean_path}")
    print(f"  Noisy: {noisy_path}")
    
    clean_lines = []
    noisy_lines = []
    
    with open(clean_path, 'r', encoding='utf-8') as f_clean, \
         open(noisy_path, 'r', encoding='utf-8') as f_noisy:
        
        for line_num, (clean, noisy) in enumerate(zip(f_clean, f_noisy)):
            clean = clean.strip()
            noisy = noisy.strip()
            
            # Filter by length
            if len(clean) < MAX_SENTENCE_LENGTH and len(noisy) < MAX_SENTENCE_LENGTH:
                clean_lines.append(clean)
                noisy_lines.append(noisy)
            
            # Stop if we've reached the limit
            if num_samples and len(clean_lines) >= num_samples:
                break
    
    print(f"✓ Loaded {len(clean_lines):,} sentence pairs")
    return clean_lines, noisy_lines


def build_character_tokenizer(texts):
    """
    Build character-level vocabulary from texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Tuple of (char_to_index, index_to_char, vocab_size)
    """
    # Find all unique characters
    all_chars = set(''.join(texts))
    chars = sorted(list(all_chars))
    
    # Build vocabulary with special tokens
    vocabulary = [PAD_TOKEN, START_TOKEN, END_TOKEN] + chars
    
    char_to_index = {char: idx for idx, char in enumerate(vocabulary)}
    index_to_char = {idx: char for idx, char in enumerate(vocabulary)}
    vocab_size = len(vocabulary)
    
    return char_to_index, index_to_char, vocab_size


def vectorize_and_pad_sequences(texts, char_to_index, max_length):
    """
    Convert text to padded sequences of character indices.
    
    Args:
        texts: List of text strings
        char_to_index: Character to index mapping
        max_length: Maximum sequence length
        
    Returns:
        Padded numpy array of shape (num_samples, max_length)
    """
    start_idx = char_to_index[START_TOKEN]
    end_idx = char_to_index[END_TOKEN]
    
    vectorized = []
    for text in texts:
        # Add START and END tokens
        indices = [start_idx] + [char_to_index.get(c, 0) for c in text] + [end_idx]
        vectorized.append(indices)
    
    # Pad sequences
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        vectorized,
        maxlen=max_length,
        padding='post',
        value=0  # PAD_TOKEN index
    )
    
    return padded


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_seq2seq_model(vocab_size, max_seq_length, embedding_dim, latent_dim):
    """
    Build encoder-decoder LSTM model for sequence-to-sequence autocorrection.
    
    Architecture:
    - Encoder: Embedding + LSTM (returns final state)
    - Decoder: Embedding + LSTM (initialized with encoder state) + Dense
    
    Args:
        vocab_size: Size of character vocabulary
        max_seq_length: Maximum sequence length
        embedding_dim: Character embedding dimension
        latent_dim: LSTM hidden state dimension
        
    Returns:
        Compiled Keras model
    """
    # Encoder
    encoder_inputs = tf.keras.layers.Input(
        shape=(max_seq_length,),
        name='encoder_input'
    )
    
    encoder_embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name='encoder_embedding'
    )(encoder_inputs)
    
    encoder_lstm = tf.keras.layers.LSTM(
        latent_dim,
        return_state=True,
        name='encoder_lstm'
    )
    
    _, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = tf.keras.layers.Input(
        shape=(max_seq_length,),
        name='decoder_input'
    )
    
    decoder_embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name='decoder_embedding'
    )(decoder_inputs)
    
    decoder_lstm = tf.keras.layers.LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name='decoder_lstm'
    )
    
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding,
        initial_state=encoder_states
    )
    
    # Output layer
    decoder_dense = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(vocab_size, activation='softmax'),
        name='output_dense'
    )
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Build model
    model = tf.keras.models.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs,
        name='autocorrect_seq2seq'
    )
    
    return model


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("GHOST TYPE CORRECTOR - MODEL TRAINING")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # ai_model directory
    data_dir = project_root / 'data'
    
    clean_path = data_dir / 'train_clean.txt'
    noisy_path = data_dir / 'train_noisy.txt'
    tokenizer_path = data_dir / 'tokenizer_config.json'
    model_path = project_root / 'autocorrect_model.h5'
    
    # Check data files exist
    if not clean_path.exists() or not noisy_path.exists():
        print("ERROR: Training data not found!")
        print("Please run 01_data_preprocessing.py first.")
        return
    
    # Load data
    print("STEP 1: Loading Training Data")
    print("-" * 70)
    clean_lines, noisy_lines = load_training_data(clean_path, noisy_path, NUM_SAMPLES)
    print()
    
    # Build tokenizer
    print("STEP 2: Building Character Tokenizer")
    print("-" * 70)
    all_text = clean_lines + noisy_lines
    char_to_index, index_to_char, vocab_size = build_character_tokenizer(all_text)
    
    print(f"✓ Vocabulary size: {vocab_size}")
    print(f"  Sample characters: {list(char_to_index.keys())[:20]}...")
    print()
    
    # Calculate max sequence length
    max_len = max(len(s) for s in all_text) + 2  # +2 for START/END tokens
    max_seq_length = min(max_len, 101)  # Cap at reasonable length
    print(f"✓ Max sequence length: {max_seq_length}")
    print()
    
    # Vectorize data
    print("STEP 3: Vectorizing and Padding Sequences")
    print("-" * 70)
    start_time = time.time()
    
    noisy_padded = vectorize_and_pad_sequences(noisy_lines, char_to_index, max_seq_length)
    clean_padded = vectorize_and_pad_sequences(clean_lines, char_to_index, max_seq_length)
    
    print(f"✓ Vectorization complete ({time.time() - start_time:.1f}s)")
    print(f"  Encoder input shape: {noisy_padded.shape}")
    print(f"  Decoder input shape: {clean_padded.shape}")
    print()
    
    # Prepare decoder targets (shifted by one position)
    decoder_target = clean_padded[:, 1:]
    padding_col = np.zeros((decoder_target.shape[0], 1), dtype=np.int32)
    decoder_target = np.concatenate([decoder_target, padding_col], axis=-1)
    print(f"  Decoder target shape: {decoder_target.shape}")
    print()
    
    # Save tokenizer configuration
    tokenizer_config = {
        'char_to_index': char_to_index,
        'index_to_char': index_to_char,
        'vocab_size': vocab_size,
        'max_seq_length': max_seq_length,
        'start_token_index': char_to_index[START_TOKEN],
        'end_token_index': char_to_index[END_TOKEN],
        'pad_token_index': char_to_index[PAD_TOKEN]
    }
    
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    print(f"✓ Tokenizer config saved: {tokenizer_path}")
    print()
    
    # Build model
    print("STEP 4: Building Model Architecture")
    print("-" * 70)
    model = build_seq2seq_model(vocab_size, max_seq_length, EMBEDDING_DIM, LATENT_DIM)
    
    print(f"✓ Model created")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  LSTM hidden units: {LATENT_DIM}")
    print()
    model.summary()
    print()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✓ Model compiled")
    print()
    
    # Train model
    print("STEP 5: Training Model")
    print("-" * 70)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Validation split: {VALIDATION_SPLIT * 100:.0f}%")
    print()
    
    start_time = time.time()
    
    history = model.fit(
        [noisy_padded, clean_padded],
        decoder_target,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print()
    print(f"✓ Training complete ({training_time:.1f}s = {training_time/60:.1f} min)")
    print()
    
    # Save model
    print("STEP 6: Saving Model")
    print("-" * 70)
    model.save(model_path)
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Model saved: {model_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print()
    
    # Training summary
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print()
    print("Next step: Run 03_model_conversion.py to convert to TensorFlow.js")
    print()


if __name__ == "__main__":
    main()
