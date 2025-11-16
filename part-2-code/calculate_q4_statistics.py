#!/usr/bin/env python3
"""
Calculate comprehensive data statistics before and after preprocessing for Q4
"""

import sys
import os
from collections import Counter
from datetime import datetime
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from transformers import T5TokenizerFast

def load_data_files():
    """Load all data files"""
    data_folder = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/data"
    
    # Original data
    with open(os.path.join(data_folder, 'train.nl'), 'r') as f:
        original_train_nl = [line.strip() for line in f if line.strip()]
    with open(os.path.join(data_folder, 'train.sql'), 'r') as f:
        original_train_sql = [line.strip() for line in f if line.strip()]
    with open(os.path.join(data_folder, 'dev.nl'), 'r') as f:
        dev_nl = [line.strip() for line in f if line.strip()]
    with open(os.path.join(data_folder, 'dev.sql'), 'r') as f:
        dev_sql = [line.strip() for line in f if line.strip()]
    
    # Preprocessed data
    preprocessed_train_nl = []
    preprocessed_train_sql = []
    
    preprocessed_nl_file = os.path.join(data_folder, 'train_preprocessed.nl')
    preprocessed_sql_file = os.path.join(data_folder, 'train_preprocessed.sql')
    
    if os.path.exists(preprocessed_nl_file) and os.path.exists(preprocessed_sql_file):
        with open(preprocessed_nl_file, 'r') as f:
            preprocessed_train_nl = [line.strip() for line in f if line.strip()]
        with open(preprocessed_sql_file, 'r') as f:
            preprocessed_train_sql = [line.strip() for line in f if line.strip()]
    
    return {
        'original_train_nl': original_train_nl,
        'original_train_sql': original_train_sql,
        'preprocessed_train_nl': preprocessed_train_nl,
        'preprocessed_train_sql': preprocessed_train_sql,
        'dev_nl': dev_nl,
        'dev_sql': dev_sql
    }

def calculate_statistics_with_tokenizer(nl_queries, sql_queries, tokenizer, name):
    """Calculate statistics using T5 tokenizer"""
    
    print(f"\n📊 Calculating statistics for {name}...")
    
    # Tokenize all queries
    nl_token_lengths = []
    sql_token_lengths = []
    nl_vocab = set()
    sql_vocab = set()
    
    for nl in nl_queries:
        tokens = tokenizer.tokenize(nl)
        nl_token_lengths.append(len(tokens))
        nl_vocab.update(tokens)
    
    for sql in sql_queries:
        tokens = tokenizer.tokenize(sql)
        sql_token_lengths.append(len(tokens))
        sql_vocab.update(tokens)
    
    # Calculate statistics
    stats = {
        'num_examples': len(nl_queries),
        'mean_nl_length': sum(nl_token_lengths) / len(nl_token_lengths) if nl_token_lengths else 0,
        'mean_sql_length': sum(sql_token_lengths) / len(sql_token_lengths) if sql_token_lengths else 0,
        'max_nl_length': max(nl_token_lengths) if nl_token_lengths else 0,
        'max_sql_length': max(sql_token_lengths) if sql_token_lengths else 0,
        'nl_vocab_size': len(nl_vocab),
        'sql_vocab_size': len(sql_vocab)
    }
    
    return stats

def print_table_1(original_train_stats, dev_stats):
    """Print Table 1: Before preprocessing"""
    
    print("\n" + "="*60)
    print("TABLE 1: Data statistics BEFORE preprocessing")
    print("="*60)
    print(f"{'Statistics Name':<30} | {'Train':<10} | {'Dev':<10}")
    print("-" * 60)
    print(f"{'Number of examples':<30} | {original_train_stats['num_examples']:<10} | {dev_stats['num_examples']:<10}")
    print(f"{'Mean sentence length':<30} | {original_train_stats['mean_nl_length']:<10.2f} | {dev_stats['mean_nl_length']:<10.2f}")
    print(f"{'Mean SQL query length':<30} | {original_train_stats['mean_sql_length']:<10.2f} | {dev_stats['mean_sql_length']:<10.2f}")
    print(f"{'Vocabulary size (natural lang)':<30} | {original_train_stats['nl_vocab_size']:<10} | {dev_stats['nl_vocab_size']:<10}")
    print(f"{'Vocabulary size (SQL)':<30} | {original_train_stats['sql_vocab_size']:<10} | {dev_stats['sql_vocab_size']:<10}")
    print("-" * 60)

def print_table_2(preprocessed_train_stats, dev_stats):
    """Print Table 2: After preprocessing"""
    
    print("\n" + "="*60)
    print("TABLE 2: Data statistics AFTER preprocessing")
    print("="*60)
    print("Model name: T5-small")
    print(f"{'Statistics Name':<30} | {'Train':<10} | {'Dev':<10}")
    print("-" * 60)
    print(f"{'Number of examples':<30} | {preprocessed_train_stats['num_examples']:<10} | {dev_stats['num_examples']:<10}")
    print(f"{'Mean sentence length':<30} | {preprocessed_train_stats['mean_nl_length']:<10.2f} | {dev_stats['mean_nl_length']:<10.2f}")
    print(f"{'Mean SQL query length':<30} | {preprocessed_train_stats['mean_sql_length']:<10.2f} | {dev_stats['mean_sql_length']:<10.2f}")
    print(f"{'Vocabulary size (natural lang)':<30} | {preprocessed_train_stats['nl_vocab_size']:<10} | {dev_stats['nl_vocab_size']:<10}")
    print(f"{'Vocabulary size (SQL)':<30} | {preprocessed_train_stats['sql_vocab_size']:<10} | {dev_stats['sql_vocab_size']:<10}")
    print("-" * 60)

def print_preprocessing_improvements(original_stats, preprocessed_stats):
    """Print improvements from preprocessing"""
    
    print("\n" + "="*60)
    print("PREPROCESSING IMPROVEMENTS")
    print("="*60)
    
    nl_length_change = preprocessed_stats['mean_nl_length'] - original_stats['mean_nl_length']
    sql_length_change = preprocessed_stats['mean_sql_length'] - original_stats['mean_sql_length']
    nl_vocab_change = preprocessed_stats['nl_vocab_size'] - original_stats['nl_vocab_size']
    sql_vocab_change = preprocessed_stats['sql_vocab_size'] - original_stats['sql_vocab_size']
    examples_change = preprocessed_stats['num_examples'] - original_stats['num_examples']
    
    print(f"Examples added through augmentation: {examples_change:+d}")
    print(f"Mean NL length change: {nl_length_change:+.2f} tokens")
    print(f"Mean SQL length change: {sql_length_change:+.2f} tokens")
    print(f"NL vocabulary change: {nl_vocab_change:+d} tokens")
    print(f"SQL vocabulary change: {sql_vocab_change:+d} tokens")

def generate_assignment_tables():
    """Generate tables formatted for the assignment"""
    
    print("\n" + "="*60)
    print("ASSIGNMENT Q4 TABLES (Copy-paste ready)")
    print("="*60)
    
    data = load_data_files()
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Calculate statistics
    original_train_stats = calculate_statistics_with_tokenizer(
        data['original_train_nl'], data['original_train_sql'], tokenizer, "Original Training"
    )
    
    dev_stats = calculate_statistics_with_tokenizer(
        data['dev_nl'], data['dev_sql'], tokenizer, "Development"
    )
    
    if data['preprocessed_train_nl'] and data['preprocessed_train_sql']:
        preprocessed_train_stats = calculate_statistics_with_tokenizer(
            data['preprocessed_train_nl'], data['preprocessed_train_sql'], tokenizer, "Preprocessed Training"
        )
    else:
        print("⚠️ No preprocessed data found. Using original data for Table 2.")
        preprocessed_train_stats = original_train_stats
    
    # Print tables
    print_table_1(original_train_stats, dev_stats)
    print_table_2(preprocessed_train_stats, dev_stats)
    
    if data['preprocessed_train_nl'] and data['preprocessed_train_sql']:
        print_preprocessing_improvements(original_train_stats, preprocessed_train_stats)
    
    # Generate clean tables for copy-paste
    print("\n" + "="*60)
    print("CLEAN TABLES FOR ASSIGNMENT")
    print("="*60)
    
    print("\nTable 1: Data statistics before any pre-processing")
    print("Statistics Name                | Train    | Dev")
    print("-" * 50)
    print(f"Number of examples             | {original_train_stats['num_examples']:<8} | {dev_stats['num_examples']:<8}")
    print(f"Mean sentence length           | {original_train_stats['mean_nl_length']:<8.2f} | {dev_stats['mean_nl_length']:<8.2f}")
    print(f"Mean SQL query length          | {original_train_stats['mean_sql_length']:<8.2f} | {dev_stats['mean_sql_length']:<8.2f}")
    print(f"Vocabulary size (natural lang) | {original_train_stats['nl_vocab_size']:<8} | {dev_stats['nl_vocab_size']:<8}")
    print(f"Vocabulary size (SQL)          | {original_train_stats['sql_vocab_size']:<8} | {dev_stats['sql_vocab_size']:<8}")
    
    print("\nTable 2: Data statistics after pre-processing")
    print("Model name: T5-small")
    print("Statistics Name                | Train    | Dev")
    print("-" * 50)
    print(f"Number of examples             | {preprocessed_train_stats['num_examples']:<8} | {dev_stats['num_examples']:<8}")
    print(f"Mean sentence length           | {preprocessed_train_stats['mean_nl_length']:<8.2f} | {dev_stats['mean_nl_length']:<8.2f}")
    print(f"Mean SQL query length          | {preprocessed_train_stats['mean_sql_length']:<8.2f} | {dev_stats['mean_sql_length']:<8.2f}")
    print(f"Vocabulary size (natural lang) | {preprocessed_train_stats['nl_vocab_size']:<8} | {dev_stats['nl_vocab_size']:<8}")
    print(f"Vocabulary size (SQL)          | {preprocessed_train_stats['sql_vocab_size']:<8} | {dev_stats['sql_vocab_size']:<8}")

def main():
    print("🔢 T5 TEXT-TO-SQL DATA STATISTICS CALCULATOR")
    print("=" * 60)
    print("Calculating comprehensive statistics using T5 tokenizer...")
    
    generate_assignment_tables()
    
    print(f"\n✅ Statistics calculation complete!")
    print(f"📝 Copy the tables above for your Q4 submission")

if __name__ == "__main__":
    main()