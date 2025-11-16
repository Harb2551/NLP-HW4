#!/usr/bin/env python3
"""
Analyze T5 tokenization for SQL and explore improvements
"""

import sys
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from transformers import T5TokenizerFast
import re
from collections import Counter

def analyze_t5_tokenization():
    """Analyze how T5 tokenizer handles SQL tokens"""
    
    print("🔍 T5 TOKENIZATION ANALYSIS FOR SQL")
    print("="*60)
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Test SQL keywords and patterns
    sql_samples = [
        "SELECT DISTINCT flight_1.flight_id FROM flight flight_1",
        "WHERE flight_1.from_airport = 'BOS' AND flight_1.to_airport = 'DEN'",
        "airport_service_1.city_code = city_1.city_code",
        "flight_1.departure_time < 900",
        "ORDER BY flight_1.departure_time",
        "COUNT(*)",
        "GROUP BY airport_1.airport_code",
        "INNER JOIN airport ON flight.from_airport = airport.airport_code"
    ]
    
    print("📊 SQL TOKENIZATION PATTERNS:")
    print("-" * 40)
    
    for sql in sql_samples:
        tokens = tokenizer.tokenize(sql)
        print(f"\nSQL: {sql}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        
        # Check for problematic patterns
        issues = []
        if any('_' in token and len(token) > 5 for token in tokens):
            issues.append("Long underscore tokens")
        if any(token.startswith('▁.') for token in tokens):
            issues.append("Dot tokenization issues")
        if len([t for t in tokens if 'flight_' in t]) > 2:
            issues.append("Repetitive alias tokens")
            
        if issues:
            print(f"⚠️  Issues: {', '.join(issues)}")

def analyze_vocabulary_coverage():
    """Check T5 vocabulary coverage for SQL keywords"""
    
    print(f"\n🔤 VOCABULARY COVERAGE ANALYSIS:")
    print("-" * 40)
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Common SQL keywords and patterns
    sql_keywords = [
        "SELECT", "DISTINCT", "FROM", "WHERE", "AND", "OR", 
        "JOIN", "INNER", "LEFT", "RIGHT", "ON", "GROUP", "BY",
        "ORDER", "HAVING", "COUNT", "SUM", "AVG", "MAX", "MIN",
        "flight_1", "airport_1", "city_1", "airline_1",
        "flight_id", "airport_code", "city_code", "departure_time"
    ]
    
    coverage_issues = []
    
    for keyword in sql_keywords:
        tokens = tokenizer.tokenize(keyword)
        if len(tokens) > 1:
            coverage_issues.append((keyword, tokens))
    
    print(f"Multi-token SQL keywords ({len(coverage_issues)} issues):")
    for keyword, tokens in coverage_issues[:10]:  # Show first 10
        print(f"  {keyword} → {tokens}")
    
    return coverage_issues

def create_sql_aware_tokenizer():
    """Create improved tokenizer for SQL"""
    
    print(f"\n⚙️ CREATING SQL-AWARE TOKENIZER:")
    print("-" * 40)
    
    base_tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # SQL-specific vocabulary to add
    sql_vocab = [
        # Common table aliases
        "flight_1", "flight_2", "airport_1", "airport_2", "city_1", "city_2",
        "airline_1", "ground_service_1", "aircraft_1", "fare_1",
        
        # Common column patterns
        "flight_id", "airport_code", "city_code", "departure_time", 
        "arrival_time", "airline_code", "aircraft_code",
        
        # SQL operators and keywords
        "SELECT_DISTINCT", "WHERE_AND", "WHERE_OR", "JOIN_ON",
        
        # Common value patterns
        "'BOS'", "'DEN'", "'ATL'", "'LAX'", "'JFK'", "'LGA'"
    ]
    
    # Check current vocabulary size
    print(f"Original vocabulary size: {len(base_tokenizer.get_vocab())}")
    
    # Add SQL tokens
    new_tokens = []
    for token in sql_vocab:
        if token not in base_tokenizer.get_vocab():
            new_tokens.append(token)
    
    print(f"Adding {len(new_tokens)} SQL-specific tokens")
    
    if new_tokens:
        base_tokenizer.add_tokens(new_tokens)
        print(f"New vocabulary size: {len(base_tokenizer.get_vocab())}")
        
        # Test improved tokenization
        test_sql = "SELECT DISTINCT flight_1.flight_id FROM flight flight_1 WHERE flight_1.airport_code = 'BOS'"
        
        print(f"\nBefore: {T5TokenizerFast.from_pretrained('google-t5/t5-small').tokenize(test_sql)}")
        print(f"After:  {base_tokenizer.tokenize(test_sql)}")
        
        return base_tokenizer
    
    return None

def analyze_schema_tokenization():
    """Analyze how schema information is tokenized"""
    
    print(f"\n🏗️ SCHEMA TOKENIZATION ANALYSIS:")
    print("-" * 40)
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Load schema information
    schema_file = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/data/flight_database.schema"
    
    try:
        with open(schema_file, 'r') as f:
            schema_content = f.read()
        
        schema_tokens = tokenizer.tokenize(schema_content)
        print(f"Schema tokens: {len(schema_tokens)}")
        print(f"Sample tokens: {schema_tokens[:20]}")
        
        # Check for inefficient tokenization
        long_tokens = [t for t in schema_tokens if len(t) > 10]
        print(f"Long tokens: {len(long_tokens)}")
        if long_tokens:
            print(f"Examples: {long_tokens[:5]}")
            
    except Exception as e:
        print(f"Could not load schema: {e}")

def suggest_tokenization_improvements():
    """Suggest specific improvements for SQL tokenization"""
    
    print(f"\n💡 TOKENIZATION IMPROVEMENT SUGGESTIONS:")
    print("-" * 40)
    
    suggestions = [
        {
            "improvement": "Add SQL-specific vocabulary",
            "description": "Pre-train tokenizer with common SQL patterns",
            "impact": "Reduce token count for table.column patterns",
            "implementation": "Add tokens like 'flight_1.flight_id', 'airport_code'"
        },
        {
            "improvement": "Subword regularization",
            "description": "Use SentencePiece with SQL corpus",
            "impact": "Better handling of database identifiers",
            "implementation": "Train custom tokenizer on SQL data"
        },
        {
            "improvement": "Special token markers",
            "description": "Add markers for table/column boundaries",
            "impact": "Help model understand SQL structure",
            "implementation": "Use <TABLE>, <COLUMN>, <ALIAS> tokens"
        },
        {
            "improvement": "Preprocessing normalization",
            "description": "Standardize SQL formatting before tokenization",
            "impact": "More consistent token patterns",
            "implementation": "Space normalization, case consistency"
        }
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion['improvement']}")
        print(f"   Description: {suggestion['description']}")
        print(f"   Expected Impact: {suggestion['impact']}")
        print(f"   Implementation: {suggestion['implementation']}\n")

def main():
    print("🚀 T5 TOKENIZATION IMPROVEMENT ANALYSIS")
    print("="*60)
    
    # Run analyses
    analyze_t5_tokenization()
    coverage_issues = analyze_vocabulary_coverage()
    improved_tokenizer = create_sql_aware_tokenizer()
    analyze_schema_tokenization()
    suggest_tokenization_improvements()
    
    print(f"\n📈 POTENTIAL F1 IMPROVEMENTS:")
    print("-" * 40)
    print("1. Reduced token count → faster training/inference")
    print("2. Better SQL pattern recognition → improved accuracy") 
    print("3. Consistent alias tokenization → fewer duplicate errors")
    print("4. Structured schema tokens → better table understanding")
    
    if coverage_issues:
        print(f"\n⚠️  Found {len(coverage_issues)} tokenization inefficiencies")
        print("Consider implementing SQL-aware tokenizer improvements")

if __name__ == "__main__":
    main()