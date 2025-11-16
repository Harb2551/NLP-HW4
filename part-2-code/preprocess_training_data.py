#!/usr/bin/env python3
"""
Training Data Preprocessing Pipeline
Apply fixes and improvements to training data based on analysis
"""

import sys
import os
import re
from typing import List, Tuple
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from schema_utils import fix_sql_syntax_errors, deduplicate_table_aliases

def preprocess_training_data():
    """Apply preprocessing improvements to training data"""
    
    print("🔧 TRAINING DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load original training data
    train_nl_file = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/data/train.nl"
    train_sql_file = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/data/train.sql"
    
    # Output files
    output_nl_file = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/data/train_preprocessed.nl"
    output_sql_file = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/data/train_preprocessed.sql"
    
    try:
        with open(train_nl_file, 'r') as f:
            original_nl = [line.strip() for line in f if line.strip()]
        
        with open(train_sql_file, 'r') as f:
            original_sql = [line.strip() for line in f if line.strip()]
            
        print(f"📂 Loaded {len(original_nl)} NL queries and {len(original_sql)} SQL queries")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Apply preprocessing steps
    processed_nl, processed_sql = apply_preprocessing_pipeline(original_nl, original_sql)
    
    # Save preprocessed data
    with open(output_nl_file, 'w') as f:
        for nl in processed_nl:
            f.write(nl + '\n')
    
    with open(output_sql_file, 'w') as f:
        for sql in processed_sql:
            f.write(sql + '\n')
    
    print(f"\n✅ Preprocessed training data saved:")
    print(f"   NL queries: {output_nl_file}")
    print(f"   SQL queries: {output_sql_file}")
    print(f"   Final size: {len(processed_nl)} query pairs")

def apply_preprocessing_pipeline(nl_queries: List[str], sql_queries: List[str]) -> Tuple[List[str], List[str]]:
    """Apply complete preprocessing pipeline"""
    
    processed_nl = []
    processed_sql = []
    
    stats = {
        'original_count': len(nl_queries),
        'nl_normalizations': 0,
        'augmented': 0
    }
    
    print(f"\n🔄 PROCESSING {len(nl_queries)} TRAINING EXAMPLES...")
    
    for i, (nl, sql) in enumerate(zip(nl_queries, sql_queries)):
        
        # 1. Normalize natural language
        processed_nl_query = normalize_natural_language(nl)
        if processed_nl_query != nl:
            stats['nl_normalizations'] += 1
        
        # 2. Keep SQL as-is (no filtering or fixing)
        processed_sql_query = sql
        
        # 3. Add the processed pair
        processed_nl.append(processed_nl_query)
        processed_sql.append(processed_sql_query)
        
        # 4. Data augmentation for important patterns
        augmented_pairs = generate_augmentations(processed_nl_query, processed_sql_query)
        for aug_nl, aug_sql in augmented_pairs:
            processed_nl.append(aug_nl)
            processed_sql.append(aug_sql)
            stats['augmented'] += 1
    
    print_preprocessing_stats(stats, len(processed_nl))
    return processed_nl, processed_sql



def normalize_natural_language(nl: str) -> str:
    """Normalize natural language queries"""
    
    # Convert to consistent case for common words
    normalized = nl
    
    # Standardize question starters
    normalized = re.sub(r'^show me ', 'list ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^get me ', 'list ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^find me ', 'list ', normalized, flags=re.IGNORECASE)
    
    # Normalize city name mentions (keep original casing in quotes)
    normalized = re.sub(r'\bdallas\b', 'DALLAS', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bdenver\b', 'DENVER', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bboston\b', 'BOSTON', normalized, flags=re.IGNORECASE)
    
    # Standardize time mentions
    normalized = re.sub(r'\b(\d{1,2}):\d{2}\b', lambda m: f"{int(m.group(1))*100}", normalized)
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized



def generate_augmentations(nl: str, sql: str) -> List[Tuple[str, str]]:
    """Generate augmented training examples - minimal augmentation"""
    
    augmentations = []
    
    # Very selective augmentation for key patterns only
    if 'flights from' in nl.lower() and 'SELECT DISTINCT flight' in sql and len(sql) < 500:
        
        # Single high-value paraphrase
        if nl.startswith('what flights'):
            paraphrase = nl.replace('what flights', 'list flights')
            if paraphrase != nl:
                augmentations.append((paraphrase, sql))
    
    return augmentations

def print_preprocessing_stats(stats: dict, final_count: int):
    """Print preprocessing statistics"""
    
    print(f"\n📊 PREPROCESSING STATISTICS:")
    print(f"   Original examples: {stats['original_count']}")
    print(f"   NL normalizations: {stats['nl_normalizations']}")
    print(f"   Augmented examples added: {stats['augmented']}")
    print(f"   Final training size: {final_count}")
    
    improvement = (final_count / stats['original_count']) * 100
    print(f"   Data expansion: {improvement:.1f}% ({final_count - stats['original_count']:+d} examples)")


def main():
    preprocess_training_data()
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Review preprocessed training data")
    print(f"   2. Update load_data.py to use train_preprocessed.* files")
    print(f"   3. Retrain model with improved data")
    print(f"   4. Expect F1 improvement from cleaner, augmented training data")

if __name__ == "__main__":
    main()