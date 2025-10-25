"""
Script 01: Data Preprocessing

Reads a raw text corpus, cleans it, splits it into train/test sets,
and generates noisy versions to create a parallel corpus for training
the autocorrect model.

This script should be run before 02_model_training.py.
"""

import os
import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import json

# --- Configuration ---
# Define paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Input file
RAW_CORPUS_PATH = os.path.join(DATA_DIR, 'raw_corpus.txt')

# Output files
CLEAN_TRAIN_PATH = os.path.join(DATA_DIR, 'train_clean.txt')
NOISY_TRAIN_PATH = os.path.join(DATA_DIR, 'train_noisy.txt')
TOKENIZER_PATH = os.path.join(DATA_DIR, 'tokenizer_config.json')

# Parameters
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
VOCAB_SIZE = 10000  # Max number of words to keep in the tokenizer
MAX_LEN = 20        # Max sequence length (must match model training)


# --- Helper Functions ---

def clean_text(text):
    """
    Applies basic text cleaning and adds special tokens.
    - Lowercase
    - Remove special characters (except basic punctuation)
    - Add <sos> (start) and <eos> (end) tokens
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,']", "", text)
    text = re.sub(r"([\w\d\s.,'])(?=\1)", "", text) # Remove consecutive duplicates
    
    # Add start and end tokens
    return f"<sos> {text} <eos>"

def add_noise(text, noise_level=0.15):
    """
    Introduces common typographical errors into a text.
    - Substitution: replace a char
    - Insertion: add a char
    - Deletion: remove a char
    - Transposition: swap adjacent chars
    """
    # Don't add noise to the special tokens
    if text.startswith("<sos> ") and text.endswith(" <eos>"):
        core_text = text[6:-6]
    else:
        core_text = text

    chars = list(core_text)
    for i in range(len(chars)):
        if random.random() < noise_level:
            noise_type = random.choice(['sub', 'ins', 'del', 'trans'])
            
            # Substitution
            if noise_type == 'sub' and chars[i].isalpha():
                chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
            
            # Insertion
            elif noise_type == 'ins' and chars[i].isalpha():
                chars.insert(i, random.choice('abcdefghijklmnopqrstuvwxyz'))
            
            # Deletion
            elif noise_type == 'del':
                chars[i] = ''
                
            # Transposition (swap with next char)
            elif noise_type == 'trans' and i < len(chars) - 1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
                
    # Re-add special tokens
    return f"<sos> {''.join(chars)} <eos>"

# --- Main Execution ---

def main():
    """
    Main preprocessing pipeline.
    """
    print("Starting data preprocessing...")

    # 1. Load raw data
    try:
        with open(RAW_CORPUS_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"Loaded {len(lines)} lines from '{RAW_CORPUS_PATH}'.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Raw corpus not found.")
        print(f"Please place your 'raw_corpus.txt' file in the '{DATA_DIR}' folder.")
        return

    # 2. Clean and filter data
    cleaned_lines = [clean_text(line) for line in lines if line.strip()]
    
    # 3. Create DataFrame
    df = pd.DataFrame({'clean': cleaned_lines})
    
    # 4. Filter by length
    df['len'] = df['clean'].apply(lambda x: len(x.split()))
    df = df[df['len'] <= MAX_LEN]
    print(f"Filtered down to {len(df)} samples with max length {MAX_LEN}.")
    
    # 5. Add noise
    print("Generating noisy data...")
    df['noisy'] = df['clean'].apply(add_noise)
    
    # 6. Split data
    train_df, test_df = train_test_split(df, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE)
    print(f"Split data into {len(train_df)} training and {len(test_df)} test samples.")
    
    # 7. Fit tokenizer
    # We fit the tokenizer on *all* data (clean and noisy) to ensure
    # it recognizes both correct words and common typos.
    all_texts = train_df['clean'].tolist() + train_df['noisy'].tolist()
    
    # We add <oov> for out-of-vocabulary words
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<oov>')
    tokenizer.fit_on_texts(all_texts)
    
    # 8. Save tokenizer
    # We must save the tokenizer as a JSON *string*
    tokenizer_json_string = tokenizer.to_json()
    with open(TOKENIZER_PATH, 'w', encoding='utf-8') as f:
        # json.dump ensures the string is saved correctly in a JSON file
        json.dump(tokenizer_json_string, f, ensure_ascii=False, indent=4)
    print(f"Tokenizer fitted and saved to '{TOKENIZER_PATH}'.")

    # 9. Save processed training files
    with open(CLEAN_TRAIN_PATH, 'w', encoding='utf-8') as f:
        f.writelines([line + '\n' for line in train_df['clean']])
        
    with open(NOISY_TRAIN_PATH, 'w', encoding='utf-8') as f:
        f.writelines([line + '\n' for line in train_df['noisy']])
        
    print(f"Saved processed training files to '{CLEAN_TRAIN_PATH}' and '{NOISY_TRAIN_PATH}'.")
    print("Data preprocessing complete.")

if __name__ == "__main__":
    main()