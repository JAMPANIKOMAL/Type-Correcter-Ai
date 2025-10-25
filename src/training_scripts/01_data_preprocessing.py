#!/usr/bin/env python3
"""
Ghost Type Corrector - Data Preprocessing Pipeline
===================================================
Prepares training data by cleaning corpus text and generating synthetic typos.

This script:
1. Loads raw text from corpus.txt
2. Cleans and normalizes the text (lowercase, remove punctuation, etc.)
3. Generates synthetic typos for training data
4. Saves paired clean/noisy datasets

Author: Ghost Type Corrector Team
License: MIT
"""

import re
import os
import random
import sys
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURATION - MAXIMUM SETTINGS
# =============================================================================

# Noise level: increased for more training variation
NOISE_LEVEL = 0.20  # Increased from 0.15 for more diverse typos

# Minimum sentence length (in words) to keep
MIN_SENTENCE_LENGTH = 3

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# TEXT CLEANING FUNCTIONS
# =============================================================================

def clean_line(text: str) -> str:
    """
    Clean a single line of text from the corpus.
    
    Transformations:
    - Remove line numbers (e.g., "1 \\t")
    - Convert to lowercase
    - Remove punctuation, symbols, and numbers
    - Normalize whitespace
    - Filter very short sentences
    
    Args:
        text: Raw text line from corpus
        
    Returns:
        Cleaned text string, or None if line should be skipped
    """
    # Remove line number prefix (e.g., "42\t")
    match = re.search(r'^\d+\t(.*)', text)
    if match:
        text = match.group(1)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove everything except letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Normalize whitespace (replace multiple spaces with single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Filter out very short sentences
    if len(text.split()) < MIN_SENTENCE_LENGTH:
        return None
    
    return text


def add_noise_to_sentence(sentence: str, noise_level: float = NOISE_LEVEL) -> str:
    """
    Add realistic typing errors to a sentence.
    
    Typo types:
    - delete: Remove a random character
    - insert: Insert a random character
    - substitute: Replace a character with another
    - swap: Swap two adjacent characters
    
    Args:
        sentence: Clean input sentence
        noise_level: Probability (0-1) that each word gets a typo
        
    Returns:
        Sentence with synthetic typos
    """
    words = sentence.split()
    noised_words = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for word in words:
        # Only add noise to longer words, with probability noise_level
        if random.random() < noise_level and len(word) > 3:
            typo_type = random.choice(['delete', 'insert', 'substitute', 'swap'])
            
            if typo_type == 'delete':
                # Remove a random character
                pos = random.randint(0, len(word) - 1)
                noised_word = word[:pos] + word[pos+1:]
                
            elif typo_type == 'insert':
                # Insert a random character
                pos = random.randint(0, len(word))
                char = random.choice(alphabet)
                noised_word = word[:pos] + char + word[pos:]
                
            elif typo_type == 'substitute':
                # Replace a character with a random one
                pos = random.randint(0, len(word) - 1)
                char = random.choice(alphabet)
                noised_word = word[:pos] + char + word[pos+1:]
                
            else:  # swap
                # Swap two adjacent characters
                if len(word) > 1:
                    pos = random.randint(0, len(word) - 2)
                    noised_word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                else:
                    noised_word = word
            
            noised_words.append(noised_word)
        else:
            # Keep original word
            noised_words.append(word)
    
    return ' '.join(noised_words)


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def main():
    """Main data preprocessing pipeline."""
    
    print("=" * 70)
    print("GHOST TYPE CORRECTOR - DATA PREPROCESSING")
    print("=" * 70)
    print()
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Noise level: {NOISE_LEVEL * 100:.1f}%")
    print()
    
    # Define file paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    input_corpus = data_dir / 'corpus.txt'
    output_clean = data_dir / 'train_clean.txt'
    output_noisy = data_dir / 'train_noisy.txt'
    
    print(f"Input:  {input_corpus}")
    print(f"Output: {output_clean}")
    print(f"        {output_noisy}")
    print()
    
    # Check if input file exists
    if not input_corpus.exists():
        print(f"ERROR: Input file not found!")
        print(f"Expected location: {input_corpus}")
        print()
        print("Please place your corpus.txt file in the ai_model/data/ directory.")
        sys.exit(1)
    
    # Count total lines for progress bar
    print("Counting lines in corpus...")
    with open(input_corpus, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Found {total_lines:,} lines")
    print()
    
    # Process corpus
    print("Processing corpus...")
    processed_count = 0
    skipped_count = 0
    
    with open(input_corpus, 'r', encoding='utf-8') as f_in, \
         open(output_clean, 'w', encoding='utf-8') as f_clean, \
         open(output_noisy, 'w', encoding='utf-8') as f_noisy:
        
        for line in tqdm(f_in, total=total_lines, desc="Processing"):
            # Clean the line
            clean_sentence = clean_line(line)
            
            if clean_sentence:
                # Generate noisy version
                noisy_sentence = add_noise_to_sentence(clean_sentence)
                
                # Write to output files
                f_clean.write(clean_sentence + '\n')
                f_noisy.write(noisy_sentence + '\n')
                
                processed_count += 1
            else:
                skipped_count += 1
    
    # Summary
    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Processed: {processed_count:,} sentences")
    print(f"Skipped:   {skipped_count:,} sentences (too short)")
    print()
    print(f"Output files created:")
    print(f"  ✓ {output_clean}")
    print(f"  ✓ {output_noisy}")
    print()
    
    # Show sample output
    print("Sample pairs (first 3):")
    print("-" * 70)
    with open(output_clean, 'r', encoding='utf-8') as f_clean, \
         open(output_noisy, 'r', encoding='utf-8') as f_noisy:
        for i in range(min(3, processed_count)):
            clean = f_clean.readline().strip()
            noisy = f_noisy.readline().strip()
            print(f"{i+1}. Clean: {clean}")
            print(f"   Noisy: {noisy}")
            print()


if __name__ == "__main__":
    main()
