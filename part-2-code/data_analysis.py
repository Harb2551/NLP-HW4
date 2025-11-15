#!/usr/bin/env python3
"""
Data Analysis Script for T5 Text-to-SQL Task
Calculates statistics needed for Q4 (Tables 1 and 2)
"""

from transformers import T5Tokenizer
import numpy as np
from collections import Counter
import os
from datetime import datetime

def load_data():
    """Load all training and dev data files"""
    print("Loading data files...")
    
    # Load training data
    with open('data/train.nl', 'r') as f:
        train_nl = [line.strip() for line in f.readlines()]
    with open('data/train.sql', 'r') as f:
        train_sql = [line.strip() for line in f.readlines()]
    
    # Load dev data  
    with open('data/dev.nl', 'r') as f:
        dev_nl = [line.strip() for line in f.readlines()]
    with open('data/dev.sql', 'r') as f:
        dev_sql = [line.strip() for line in f.readlines()]
    
    return train_nl, train_sql, dev_nl, dev_sql

def calculate_statistics(train_nl, train_sql, dev_nl, dev_sql, tokenizer):
    """Calculate all required statistics"""
    
    print("Calculating statistics...")
    
    # Basic counts
    stats = {}
    stats['train_examples'] = len(train_nl)
    stats['dev_examples'] = len(dev_nl)
    
    # Tokenize all data
    print("Tokenizing natural language queries...")
    train_nl_tokens = [tokenizer.tokenize(text) for text in train_nl]
    dev_nl_tokens = [tokenizer.tokenize(text) for text in dev_nl]
    
    print("Tokenizing SQL queries...")
    train_sql_tokens = [tokenizer.tokenize(text) for text in train_sql]
    dev_sql_tokens = [tokenizer.tokenize(text) for text in dev_sql]
    
    # Calculate mean lengths
    stats['train_nl_mean_length'] = np.mean([len(tokens) for tokens in train_nl_tokens])
    stats['train_sql_mean_length'] = np.mean([len(tokens) for tokens in train_sql_tokens])
    stats['dev_nl_mean_length'] = np.mean([len(tokens) for tokens in dev_nl_tokens])
    stats['dev_sql_mean_length'] = np.mean([len(tokens) for tokens in dev_sql_tokens])
    
    # Calculate vocabulary sizes
    print("Calculating vocabulary sizes...")
    
    # Flatten all tokens
    all_train_nl_tokens = []
    all_train_sql_tokens = []
    all_dev_nl_tokens = []
    all_dev_sql_tokens = []
    
    for tokens in train_nl_tokens:
        all_train_nl_tokens.extend(tokens)
    for tokens in train_sql_tokens:
        all_train_sql_tokens.extend(tokens)
    for tokens in dev_nl_tokens:
        all_dev_nl_tokens.extend(tokens)
    for tokens in dev_sql_tokens:
        all_dev_sql_tokens.extend(tokens)
    
    # Count unique tokens
    stats['train_nl_vocab_size'] = len(set(all_train_nl_tokens))
    stats['train_sql_vocab_size'] = len(set(all_train_sql_tokens))
    stats['dev_nl_vocab_size'] = len(set(all_dev_nl_tokens))
    stats['dev_sql_vocab_size'] = len(set(all_dev_sql_tokens))
    
    # Combined vocabulary sizes
    all_nl_tokens = all_train_nl_tokens + all_dev_nl_tokens
    all_sql_tokens = all_train_sql_tokens + all_dev_sql_tokens
    stats['combined_nl_vocab_size'] = len(set(all_nl_tokens))
    stats['combined_sql_vocab_size'] = len(set(all_sql_tokens))
    
    # Additional useful statistics
    stats['train_nl_max_length'] = max([len(tokens) for tokens in train_nl_tokens])
    stats['train_sql_max_length'] = max([len(tokens) for tokens in train_sql_tokens])
    stats['dev_nl_max_length'] = max([len(tokens) for tokens in dev_nl_tokens])
    stats['dev_sql_max_length'] = max([len(tokens) for tokens in dev_sql_tokens])
    
    return stats

def save_statistics(stats, output_file='data_statistics.txt'):
    """Save statistics to a text file"""
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("T5 TEXT-TO-SQL DATA ANALYSIS REPORT\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        # Basic Statistics
        f.write("BASIC STATISTICS:\n")
        f.write("-"*20 + "\n")
        f.write(f"Training examples: {stats['train_examples']}\n")
        f.write(f"Development examples: {stats['dev_examples']}\n")
        f.write(f"Total examples: {stats['train_examples'] + stats['dev_examples']}\n\n")
        
        # Mean Lengths
        f.write("MEAN SEQUENCE LENGTHS (in tokens):\n")
        f.write("-"*35 + "\n")
        f.write(f"Training NL mean length: {stats['train_nl_mean_length']:.2f}\n")
        f.write(f"Training SQL mean length: {stats['train_sql_mean_length']:.2f}\n")
        f.write(f"Dev NL mean length: {stats['dev_nl_mean_length']:.2f}\n")
        f.write(f"Dev SQL mean length: {stats['dev_sql_mean_length']:.2f}\n\n")
        
        # Max Lengths
        f.write("MAXIMUM SEQUENCE LENGTHS (in tokens):\n")
        f.write("-"*37 + "\n")
        f.write(f"Training NL max length: {stats['train_nl_max_length']}\n")
        f.write(f"Training SQL max length: {stats['train_sql_max_length']}\n")
        f.write(f"Dev NL max length: {stats['dev_nl_max_length']}\n")
        f.write(f"Dev SQL max length: {stats['dev_sql_max_length']}\n\n")
        
        # Vocabulary Sizes
        f.write("VOCABULARY SIZES:\n")
        f.write("-"*17 + "\n")
        f.write(f"Training NL vocabulary: {stats['train_nl_vocab_size']}\n")
        f.write(f"Training SQL vocabulary: {stats['train_sql_vocab_size']}\n")
        f.write(f"Dev NL vocabulary: {stats['dev_nl_vocab_size']}\n")
        f.write(f"Dev SQL vocabulary: {stats['dev_sql_vocab_size']}\n")
        f.write(f"Combined NL vocabulary: {stats['combined_nl_vocab_size']}\n")
        f.write(f"Combined SQL vocabulary: {stats['combined_sql_vocab_size']}\n\n")
        
        # Tables for Assignment
        f.write("="*60 + "\n")
        f.write("TABLES FOR ASSIGNMENT Q4\n")
        f.write("="*60 + "\n\n")
        
        # Table 1: Before preprocessing
        f.write("Table 1: Data statistics BEFORE preprocessing\n")
        f.write("-"*45 + "\n")
        f.write("Statistics Name                | Train    | Dev\n")
        f.write("-"*45 + "\n")
        f.write(f"Number of examples             | {stats['train_examples']:<8} | {stats['dev_examples']}\n")
        f.write(f"Mean sentence length           | {stats['train_nl_mean_length']:<8.2f} | {stats['dev_nl_mean_length']:.2f}\n")
        f.write(f"Mean SQL query length          | {stats['train_sql_mean_length']:<8.2f} | {stats['dev_sql_mean_length']:.2f}\n")
        f.write(f"Vocabulary size (natural lang) | {stats['train_nl_vocab_size']:<8} | {stats['dev_nl_vocab_size']}\n")
        f.write(f"Vocabulary size (SQL)          | {stats['train_sql_vocab_size']:<8} | {stats['dev_sql_vocab_size']}\n")
        f.write("-"*45 + "\n\n")
        
        # Table 2: After preprocessing (placeholder)
        f.write("Table 2: Data statistics AFTER preprocessing\n")
        f.write("-"*44 + "\n")
        f.write("Model name: T5-small\n")
        f.write("Statistics Name                | Train    | Dev\n")
        f.write("-"*44 + "\n")
        f.write("(These will be filled after implementing preprocessing)\n")
        f.write("Mean sentence length           | TBD      | TBD\n")
        f.write("Mean SQL query length          | TBD      | TBD\n") 
        f.write("Vocabulary size (natural lang) | TBD      | TBD\n")
        f.write("Vocabulary size (SQL)          | TBD      | TBD\n")
        f.write("-"*44 + "\n\n")
        
        # Sample data
        f.write("SAMPLE DATA EXAMPLES:\n")
        f.write("-"*21 + "\n")
        f.write("(First 3 training examples will be added here)\n\n")

def main():
    """Main function to run the data analysis"""
    print("="*60)
    print("T5 TEXT-TO-SQL DATA ANALYSIS")
    print("="*60)
    
    # Load tokenizer
    print("Loading T5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    # Load data
    train_nl, train_sql, dev_nl, dev_sql = load_data()
    
    # Verify data integrity
    assert len(train_nl) == len(train_sql), f"Train NL-SQL mismatch: {len(train_nl)} vs {len(train_sql)}"
    assert len(dev_nl) == len(dev_sql), f"Dev NL-SQL mismatch: {len(dev_nl)} vs {len(dev_sql)}"
    print(f"✓ Data integrity check passed")
    
    # Calculate statistics
    stats = calculate_statistics(train_nl, train_sql, dev_nl, dev_sql, tokenizer)
    
    # Save to file
    output_file = 'data_statistics.txt'
    save_statistics(stats, output_file)
    
    print(f"\n✓ Analysis complete! Results saved to: {output_file}")
    print("\nQuick Summary:")
    print(f"  - Training examples: {stats['train_examples']}")
    print(f"  - Dev examples: {stats['dev_examples']}")
    print(f"  - Mean NL length: {stats['train_nl_mean_length']:.1f} tokens")
    print(f"  - Mean SQL length: {stats['train_sql_mean_length']:.1f} tokens")
    print(f"  - NL vocabulary: {stats['combined_nl_vocab_size']} unique tokens")
    print(f"  - SQL vocabulary: {stats['combined_sql_vocab_size']} unique tokens")

if __name__ == "__main__":
    main()