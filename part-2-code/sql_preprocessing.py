"""
Removed: SQL preprocessing utilities used only for the custom tokenizer.
"""

raise SystemExit("sql_preprocessing.py removed as part of reverting custom tokenizer changes.")
#!/usr/bin/env python3
"""
SQL preprocessing utilities for optimized tokenization
"""

import re

def preprocess_sql_for_tokenization(sql_query):
    """
    Preprocess SQL queries for better tokenization
    """
    
    # Start with the original query
    processed = sql_query
    
    # 1. Standardize spacing around operators
    processed = re.sub(r'\s*=\s*', ' = ', processed)
    processed = re.sub(r'\s*<\s*', ' < ', processed)
    processed = re.sub(r'\s*>\s*', ' > ', processed)
    processed = re.sub(r'\s*<=\s*', ' <= ', processed)
    processed = re.sub(r'\s*>=\s*', ' >= ', processed)
    processed = re.sub(r'\s*!=\s*', ' != ', processed)
    
    # 2. Normalize SQL keywords for better tokenization
    processed = re.sub(r'\bSELECT\s+DISTINCT\b', 'SELECT_DISTINCT', processed, flags=re.IGNORECASE)
    processed = re.sub(r'\bGROUP\s+BY\b', 'GROUP_BY', processed, flags=re.IGNORECASE)
    processed = re.sub(r'\bORDER\s+BY\b', 'ORDER_BY', processed, flags=re.IGNORECASE)
    processed = re.sub(r'\bINNER\s+JOIN\b', 'INNER_JOIN', processed, flags=re.IGNORECASE)
    processed = re.sub(r'\bLEFT\s+JOIN\b', 'LEFT_JOIN', processed, flags=re.IGNORECASE)
    
    # 3. Standardize common patterns
    processed = re.sub(r'\b1\s*=\s*1\b', '1=1', processed)
    processed = re.sub(r'\bAND\s+1\s*=\s*1\b', 'AND_1=1', processed)
    processed = re.sub(r'\bWHERE\s+1\s*=\s*1\b', 'WHERE_1=1', processed)
    
    # 4. Normalize whitespace
    processed = re.sub(r'\s+', ' ', processed)
    processed = processed.strip()
    
    return processed

def preprocess_nl_for_tokenization(nl_query):
    """
    Preprocess natural language queries for consistency
    """
    
    processed = nl_query
    
    # Standardize common question patterns
    processed = re.sub(r'^show me ', 'list ', processed, flags=re.IGNORECASE)
    processed = re.sub(r'^get me ', 'list ', processed, flags=re.IGNORECASE)
    processed = re.sub(r'^find me ', 'list ', processed, flags=re.IGNORECASE)
    
    # Normalize city names
    processed = re.sub(r'\bdallas\b', 'DALLAS', processed, flags=re.IGNORECASE)
    processed = re.sub(r'\bdenver\b', 'DENVER', processed, flags=re.IGNORECASE)
    processed = re.sub(r'\bboston\b', 'BOSTON', processed, flags=re.IGNORECASE)
    processed = re.sub(r'\batlanta\b', 'ATLANTA', processed, flags=re.IGNORECASE)
    
    # Normalize whitespace
    processed = ' '.join(processed.split())
    
    return processed