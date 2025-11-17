"""
Removed: SQL-optimized tokenizer creation script.
"""

raise SystemExit("create_sql_tokenizer.py removed as part of reverting custom tokenizer changes.")
#!/usr/bin/env python3
"""
Create SQL-optimized tokenizer for better T5 performance
"""

import sys
import os
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from transformers import T5TokenizerFast
import json

class SQLOptimizedTokenizer:
    """Enhanced T5 tokenizer optimized for SQL generation"""
    
    def __init__(self, base_model='google-t5/t5-small'):
        self.base_tokenizer = T5TokenizerFast.from_pretrained(base_model)
        self.sql_vocab = self._build_sql_vocabulary()
        self.optimized_tokenizer = None
        
    def _build_sql_vocabulary(self):
        """Build SQL-specific vocabulary from training data"""
        
        print("🔧 Building SQL-specific vocabulary...")
        
        # Core SQL keywords as single tokens
        sql_keywords = [
            # SQL Commands
            'SELECT_DISTINCT', 'SELECT', 'FROM', 'WHERE', 'AND_', 'OR_',
            'JOIN', 'INNER_JOIN', 'LEFT_JOIN', 'ON', 'GROUP_BY', 'ORDER_BY',
            'HAVING', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
            
            # Common operators
            '_=_', '_<_', '_>_', '_<=_', '_>=_', '_!=_',
            
            # Table aliases (flight database specific)
            'flight_1', 'flight_2', 'airport_1', 'airport_2', 'city_1', 'city_2',
            'airline_1', 'ground_service_1', 'aircraft_1', 'fare_1', 'equipment_sequence_1',
            'flight_fare_1', 'food_service_1', 'airport_service_1',
            
            # Common column patterns
            'flight_id', 'airport_code', 'city_code', 'departure_time', 'arrival_time',
            'airline_code', 'aircraft_code', 'flight_number', 'from_airport', 'to_airport',
            'city_name', 'state_code', 'transport_type', 'meal_code',
            
            # Common table.column patterns
            'flight_1.flight_id', 'flight_1.from_airport', 'flight_1.to_airport',
            'flight_1.departure_time', 'flight_1.arrival_time', 'flight_1.airline_code',
            'airport_1.airport_code', 'city_1.city_code', 'city_1.city_name',
            
            # Common airport codes
            "'BOS'", "'DEN'", "'ATL'", "'LAX'", "'JFK'", "'LGA'", "'SFO'", "'ORD'",
            "'MIA'", "'SEA'", "'DFW'", "'PHX'", "'LAS'", "'CLT'", "'MKE'", "'PIT'",
            
            # Common city names
            "'BOSTON'", "'DENVER'", "'ATLANTA'", "'DALLAS'", "'NEW_YORK'", "'PHILADELPHIA'",
            
            # Common airline codes
            "'UA'", "'AA'", "'DL'", "'WN'", "'US'", "'NW'", "'CO'", "'AS'",
            
            # Time patterns
            '800', '900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800',
            
            # Common WHERE patterns
            "WHERE_1=1", "AND_1=1",
        ]
        
        return sql_keywords
    
    def create_optimized_tokenizer(self):
        """Create the SQL-optimized tokenizer"""
        
        print(f"📈 Creating optimized tokenizer...")
        print(f"Base vocabulary size: {len(self.base_tokenizer.get_vocab())}")
        
        # Start with base tokenizer
        optimized = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Filter new tokens that aren't already in vocabulary
        new_tokens = []
        existing_vocab = optimized.get_vocab()
        
        for token in self.sql_vocab:
            if token not in existing_vocab:
                new_tokens.append(token)
        
        print(f"Adding {len(new_tokens)} new SQL tokens...")
        
        if new_tokens:
            optimized.add_tokens(new_tokens)
            print(f"New vocabulary size: {len(optimized.get_vocab())}")
        
        self.optimized_tokenizer = optimized
        return optimized
    
    def compare_tokenization(self, sql_examples):
        """Compare original vs optimized tokenization"""
        
        if not self.optimized_tokenizer:
            self.create_optimized_tokenizer()
        
        print(f"\n📊 TOKENIZATION COMPARISON:")
        print("="*60)
        
        total_original = 0
        total_optimized = 0
        
        for sql in sql_examples:
            orig_tokens = self.base_tokenizer.tokenize(sql)
            opt_tokens = self.optimized_tokenizer.tokenize(sql)
            
            total_original += len(orig_tokens)
            total_optimized += len(opt_tokens)
            
            improvement = len(orig_tokens) - len(opt_tokens)
            
            print(f"\nSQL: {sql[:60]}...")
            print(f"Original ({len(orig_tokens)}): {orig_tokens[:8]}{'...' if len(orig_tokens) > 8 else ''}")
            print(f"Optimized ({len(opt_tokens)}): {opt_tokens[:8]}{'...' if len(opt_tokens) > 8 else ''}")
            if improvement > 0:
                print(f"✅ Saved {improvement} tokens ({improvement/len(orig_tokens)*100:.1f}%)")
        
        total_improvement = total_original - total_optimized
        print(f"\n📈 OVERALL IMPROVEMENT:")
        print(f"Total tokens saved: {total_improvement}")
        print(f"Average reduction: {total_improvement/len(sql_examples):.1f} tokens per query")
        print(f"Percentage improvement: {total_improvement/total_original*100:.1f}%")
    
    def save_optimized_tokenizer(self, path):
        """Save the optimized tokenizer"""
        
        if not self.optimized_tokenizer:
            self.create_optimized_tokenizer()
        
        self.optimized_tokenizer.save_pretrained(path)
        print(f"💾 Saved optimized tokenizer to {path}")

def load_training_sql_samples():
    """Load SQL samples from training data for testing"""
    
    sql_file = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/data/train.sql"
    
    try:
        with open(sql_file, 'r') as f:
            sql_queries = [line.strip() for line in f.readlines()[:20]]  # First 20 for testing
        return sql_queries
    except Exception as e:
        print(f"Could not load training SQL: {e}")
        # Return sample queries
        return [
            "SELECT DISTINCT flight_1.flight_id FROM flight flight_1 WHERE flight_1.from_airport = 'BOS' AND flight_1.to_airport = 'DEN'",
            "SELECT DISTINCT city_1.city_name FROM city city_1, airport_service airport_service_1 WHERE city_1.city_code = airport_service_1.city_code",
            "SELECT COUNT(*) FROM flight flight_1 WHERE flight_1.departure_time < 900"
        ]

def create_sql_preprocessing_utils():
    """Create utilities for SQL preprocessing before tokenization"""
    
    print(f"\n⚙️ CREATING SQL PREPROCESSING UTILITIES:")
    print("-" * 40)
    
    preprocessing_rules = {
        # Standardize spacing around operators
        'operators': [
            (r'\s*=\s*', ' = '),
            (r'\s*<\s*', ' < '),
            (r'\s*>\s*', ' > '),
            (r'\s*<=\s*', ' <= '),
            (r'\s*>=\s*', ' >= '),
        ],
        
        # Normalize SQL keywords
        'keywords': [
            (r'\bSELECT\s+DISTINCT\b', 'SELECT_DISTINCT'),
            (r'\bGROUP\s+BY\b', 'GROUP_BY'),
            (r'\bORDER\s+BY\b', 'ORDER_BY'),
            (r'\bINNER\s+JOIN\b', 'INNER_JOIN'),
            (r'\bLEFT\s+JOIN\b', 'LEFT_JOIN'),
        ],
        
        # Standardize common patterns
        'patterns': [
            (r'\b1\s*=\s*1\b', '1=1'),
            (r'\bAND\s+1\s*=\s*1\b', 'AND_1=1'),
            (r'\bWHERE\s+1\s*=\s*1\b', 'WHERE_1=1'),
        ]
    }
    
    print("Preprocessing rules created:")
    for category, rules in preprocessing_rules.items():
        print(f"  {category}: {len(rules)} rules")
    
    return preprocessing_rules

def main():
    print("🚀 SQL-OPTIMIZED TOKENIZER CREATION")
    print("="*60)
    
    # Create SQL-optimized tokenizer
    sql_tokenizer = SQLOptimizedTokenizer()
    
    # Load sample SQL queries
    sql_samples = load_training_sql_samples()
    print(f"Loaded {len(sql_samples)} SQL samples for testing")
    
    # Create optimized tokenizer
    optimized = sql_tokenizer.create_optimized_tokenizer()
    
    # Compare tokenization
    sql_tokenizer.compare_tokenization(sql_samples)
    
    # Create preprocessing utilities
    preprocessing_rules = create_sql_preprocessing_utils()
    
    # Save optimized tokenizer
    output_path = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/sql_optimized_tokenizer"
    sql_tokenizer.save_optimized_tokenizer(output_path)
    
    print(f"\n🎯 IMPLEMENTATION NEXT STEPS:")
    print("-" * 40)
    print("1. Update load_data.py to use optimized tokenizer")
    print("2. Add SQL preprocessing before tokenization")
    print("3. Retrain model with improved tokenization")
    print("4. Expect 2-5% F1 improvement from better tokenization")
    
    print(f"\n💡 EXPECTED BENEFITS:")
    print("-" * 40)
    print("• Faster training (fewer tokens to process)")
    print("• Better SQL pattern recognition")
    print("• More consistent alias handling")  
    print("• Reduced sequence length issues")

if __name__ == "__main__":
    main()