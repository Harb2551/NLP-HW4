#!/usr/bin/env python3
"""
Training Data Analysis for Text-to-SQL Preprocessing Improvements
Analyze training data patterns to identify preprocessing opportunities
"""

import sys
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from utils import read_queries

def analyze_training_data():
    """Comprehensive analysis of training data quality and patterns"""
    
    # Load training data
    train_nl_file = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/data/train.nl"
    train_sql_file = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/data/train.sql"
    
    print("🔍 TRAINING DATA ANALYSIS FOR PREPROCESSING IMPROVEMENTS")
    print("="*70)
    
    try:
        with open(train_nl_file, 'r') as f:
            train_nl = [line.strip() for line in f if line.strip()]
        
        train_sql = read_queries(train_sql_file)
        
        print(f"📊 Dataset Size: {len(train_nl)} NL queries, {len(train_sql)} SQL queries")
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return
    
    # 1. SQL Quality Analysis
    print(f"\n1️⃣ SQL QUALITY ANALYSIS")
    print("-" * 50)
    analyze_sql_quality(train_sql)
    
    # 2. Natural Language Pattern Analysis  
    print(f"\n2️⃣ NATURAL LANGUAGE PATTERNS")
    print("-" * 50)
    analyze_nl_patterns(train_nl)
    
    # 3. Schema Usage Analysis
    print(f"\n3️⃣ SCHEMA USAGE PATTERNS")
    print("-" * 50)
    analyze_schema_usage(train_sql)
    
    # 4. Complexity Analysis
    print(f"\n4️⃣ QUERY COMPLEXITY ANALYSIS")
    print("-" * 50)
    analyze_query_complexity(train_sql)
    
    # 5. Error Pattern Detection
    print(f"\n5️⃣ POTENTIAL TRAINING DATA ISSUES")
    print("-" * 50)
    detect_training_issues(train_nl, train_sql)
    
    # 6. Preprocessing Recommendations
    print(f"\n6️⃣ PREPROCESSING RECOMMENDATIONS")
    print("-" * 50)
    generate_preprocessing_recommendations(train_nl, train_sql)

def analyze_sql_quality(sql_queries: List[str]):
    """Analyze SQL query quality and consistency"""
    
    syntax_issues = 0
    formatting_issues = 0
    alias_inconsistencies = 0
    
    for i, sql in enumerate(sql_queries[:100]):  # Sample first 100
        # Check for syntax issues
        if sql.count('(') != sql.count(')'):
            syntax_issues += 1
        
        # Check formatting consistency
        if not sql.strip().startswith('SELECT'):
            formatting_issues += 1
        
        # Check alias patterns
        aliases = re.findall(r'\b(\w+_\d+)\b', sql)
        if len(aliases) != len(set(aliases)):
            alias_inconsistencies += 1
    
    print(f"   Syntax issues (sample): {syntax_issues}/100 ({syntax_issues}%)")
    print(f"   Formatting issues: {formatting_issues}/100 ({formatting_issues}%)")
    print(f"   Alias inconsistencies: {alias_inconsistencies}/100 ({alias_inconsistencies}%)")
    
    # Analyze SQL patterns
    select_patterns = Counter()
    where_patterns = Counter()
    
    for sql in sql_queries[:200]:  # Larger sample
        # SELECT patterns
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE)
        if select_match:
            select_patterns[select_match.group(1).strip()] += 1
        
        # WHERE clause complexity
        where_count = sql.upper().count('WHERE')
        and_count = sql.upper().count(' AND ')
        or_count = sql.upper().count(' OR ')
        
        complexity = where_count + and_count + or_count
        if complexity > 5:
            where_patterns['complex'] += 1
        elif complexity > 2:
            where_patterns['medium'] += 1
        else:
            where_patterns['simple'] += 1
    
    print(f"\n   Common SELECT patterns:")
    for pattern, count in select_patterns.most_common(5):
        print(f"     {pattern}: {count} times")
    
    print(f"\n   WHERE clause complexity:")
    for complexity, count in where_patterns.most_common():
        print(f"     {complexity}: {count} queries")

def analyze_nl_patterns(nl_queries: List[str]):
    """Analyze natural language query patterns"""
    
    # Common phrases and patterns
    phrases = Counter()
    question_words = Counter()
    
    for nl in nl_queries:
        nl_lower = nl.lower()
        
        # Extract question words
        words = nl_lower.split()
        if words:
            first_word = words[0]
            if first_word in ['show', 'list', 'find', 'get', 'what', 'which', 'how', 'when', 'where']:
                question_words[first_word] += 1
        
        # Common phrases
        common_phrases = [
            'show me', 'list all', 'find all', 'get all', 'how many',
            'what is', 'which flight', 'flights from', 'flights to',
            'departure time', 'arrival time', 'flight number'
        ]
        
        for phrase in common_phrases:
            if phrase in nl_lower:
                phrases[phrase] += 1
    
    print(f"   Most common question starters:")
    for word, count in question_words.most_common(10):
        print(f"     '{word}': {count} times")
    
    print(f"\n   Most common phrases:")
    for phrase, count in phrases.most_common(10):
        print(f"     '{phrase}': {count} times")
    
    # Length analysis
    lengths = [len(nl.split()) for nl in nl_queries]
    avg_length = sum(lengths) / len(lengths)
    print(f"\n   Average NL query length: {avg_length:.1f} words")
    print(f"   Min length: {min(lengths)} words")
    print(f"   Max length: {max(lengths)} words")

def analyze_schema_usage(sql_queries: List[str]):
    """Analyze how database schema is used in training"""
    
    table_usage = Counter()
    column_usage = Counter()
    join_patterns = Counter()
    
    for sql in sql_queries:
        # Extract table usage
        tables = re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        table_usage.update(tables)
        
        # Extract column usage
        columns = re.findall(r'(\w+\.\w+)', sql)
        column_usage.update(columns)
        
        # Count joins
        join_count = len(re.findall(r'(\w+\s+\w+_\d+)', sql))
        if join_count > 5:
            join_patterns['many_joins'] += 1
        elif join_count > 2:
            join_patterns['some_joins'] += 1
        else:
            join_patterns['few_joins'] += 1
    
    print(f"   Most used tables:")
    for table, count in table_usage.most_common(10):
        print(f"     {table}: {count} times")
    
    print(f"\n   Most used columns:")
    for column, count in column_usage.most_common(10):
        print(f"     {column}: {count} times")
    
    print(f"\n   Join complexity distribution:")
    for pattern, count in join_patterns.most_common():
        print(f"     {pattern}: {count} queries")

def analyze_query_complexity(sql_queries: List[str]):
    """Analyze SQL query complexity patterns"""
    
    complexity_stats = {
        'subqueries': 0,
        'aggregations': 0,
        'multiple_conditions': 0,
        'time_conditions': 0,
        'city_conditions': 0
    }
    
    for sql in sql_queries[:500]:  # Sample
        sql_upper = sql.upper()
        
        if '(' in sql and 'SELECT' in sql[sql.find('('):]:
            complexity_stats['subqueries'] += 1
        
        if any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']):
            complexity_stats['aggregations'] += 1
        
        if sql_upper.count(' AND ') > 2:
            complexity_stats['multiple_conditions'] += 1
        
        if 'time' in sql.lower():
            complexity_stats['time_conditions'] += 1
        
        if 'city' in sql.lower():
            complexity_stats['city_conditions'] += 1
    
    print(f"   Complexity patterns (sample of 500):")
    for pattern, count in complexity_stats.items():
        percentage = (count / 500) * 100
        print(f"     {pattern}: {count} ({percentage:.1f}%)")

def detect_training_issues(nl_queries: List[str], sql_queries: List[str]):
    """Detect potential issues in training data"""
    
    issues = defaultdict(list)
    
    for i, (nl, sql) in enumerate(zip(nl_queries[:200], sql_queries[:200])):
        # Check for mismatched lengths
        if len(nl.split()) < 3:
            issues['short_nl'].append(i)
        
        if len(sql) < 20:
            issues['short_sql'].append(i)
        
        # Check for inconsistent patterns
        if 'flight' in nl.lower() and 'flight' not in sql.lower():
            issues['missing_flight_table'].append(i)
        
        if 'city' in nl.lower() and 'city' not in sql.lower():
            issues['missing_city_table'].append(i)
        
        # Check for syntax issues
        if sql.count('(') != sql.count(')'):
            issues['unbalanced_parens'].append(i)
        
        # Check for encoding issues
        if any(ord(c) > 127 for c in nl + sql):
            issues['encoding_issues'].append(i)
    
    print(f"   Potential issues found (sample of 200):")
    for issue_type, indices in issues.items():
        print(f"     {issue_type}: {len(indices)} instances")
        if indices:
            print(f"       Example indices: {indices[:5]}")

def generate_preprocessing_recommendations(nl_queries: List[str], sql_queries: List[str]):
    """Generate specific preprocessing recommendations"""
    
    recommendations = []
    
    # 1. Consistency fixes
    recommendations.append("1. CONSISTENCY NORMALIZATION:")
    recommendations.append("   - Normalize question starters: 'show me' -> 'list'")
    recommendations.append("   - Standardize city names: 'DENVER' vs 'Denver'")
    recommendations.append("   - Consistent time formats: '900' vs '9:00'")
    
    # 2. SQL cleaning
    recommendations.append("\n2. SQL QUERY CLEANING:")
    recommendations.append("   - Apply same post-processing fixes to training SQL")
    recommendations.append("   - Fix alias duplication in training data")
    recommendations.append("   - Standardize spacing and formatting")
    
    # 3. Data augmentation
    recommendations.append("\n3. DATA AUGMENTATION:")
    recommendations.append("   - Add paraphrases for common questions")
    recommendations.append("   - Include negative examples (impossible queries)")
    recommendations.append("   - Add more time-based query variations")
    
    # 4. Schema enhancement
    recommendations.append("\n4. SCHEMA ENHANCEMENT:")
    recommendations.append("   - Include more detailed schema info in inputs")
    recommendations.append("   - Add foreign key relationship information")
    recommendations.append("   - Include sample column values")
    
    # 5. Quality filtering
    recommendations.append("\n5. QUALITY FILTERING:")
    recommendations.append("   - Remove queries with syntax errors")
    recommendations.append("   - Filter out overly complex queries (>8 tables)")
    recommendations.append("   - Remove duplicate NL-SQL pairs")
    
    for rec in recommendations:
        print(rec)
    
    print(f"\n💡 PRIORITY RECOMMENDATIONS:")
    print(f"   1. Apply post-processing fixes to training SQL queries")
    print(f"   2. Normalize city names and time formats")
    print(f"   3. Add paraphrases for top 10 question patterns")
    print(f"   4. Filter out syntax-error examples")
    print(f"   5. Enhance schema information in training inputs")

def main():
    analyze_training_data()

if __name__ == "__main__":
    main()