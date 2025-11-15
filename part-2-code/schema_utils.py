"""
Schema utilities for T5 text-to-SQL model.

This module contains functions for extracting and formatting database schema
information to enhance the model input.
"""

import sqlite3
from typing import Dict, List, Tuple

DB_PATH = 'data/flight_database.db'

def get_database_schema() -> Dict[str, List[Tuple[str, str]]]:
    """
    Extract database schema information.
    
    Returns:
        Dict mapping table names to list of (column_name, column_type) tuples
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        # Extract (column_name, column_type) pairs
        schema[table] = [(col[1], col[2]) for col in columns]
    
    conn.close()
    return schema

def format_schema_compact() -> str:
    """
    Format schema in a compact way suitable for T5 input.
    
    Returns:
        Compact schema string
    """
    schema = get_database_schema()
    
    # Create a concise schema representation
    schema_parts = []
    
    # Key tables for flight queries
    key_tables = ['flight', 'city', 'airport', 'airport_service', 'airline', 'aircraft']
    
    for table in key_tables:
        if table in schema:
            # Get most important columns (first 6 to keep it manageable)
            cols = schema[table][:6]
            col_names = [col[0] for col in cols]
            schema_parts.append(f"{table}({', '.join(col_names)})")
    
    return "Schema: " + " | ".join(schema_parts)

def format_enhanced_input(natural_language: str) -> str:
    """
    Create enhanced input with schema information and proper formatting.
    
    Args:
        natural_language: The original natural language query
    
    Returns:
        Enhanced input string with schema and formatting
    """
    schema_info = format_schema_compact()
    
    enhanced_input = f"""translate English to SQL:
{schema_info}
Question: {natural_language.strip()}
Answer:"""
    
    return enhanced_input

def format_enhanced_target(sql_query: str) -> str:
    """
    Format the target SQL with consistent structure and END token.
    
    Args:
        sql_query: The target SQL query
    
    Returns:
        Formatted target string with <END> token
    """
    return sql_query.strip() + " <END>"

def extract_sql_from_output(generated_output: str) -> str:
    """
    Extract SQL query from generated output by finding <END> token.
    
    Args:
        generated_output: The raw generated text from the model
    
    Returns:
        Extracted SQL query without the <END> token
    """
    output = generated_output.strip()
    
    # Look for <END> token
    if "<END>" in output:
        sql = output.split("<END>")[0].strip()
    else:
        # Fallback: use the entire output if no <END> token found
        sql = output
    
    return sql

if __name__ == "__main__":
    # Test the schema extraction
    print("=== Database Schema ===")
    schema = get_database_schema()
    for table, cols in list(schema.items())[:5]:  # Show first 5 tables
        print(f"\n{table}:")
        for col_name, col_type in cols:
            print(f"  {col_name} ({col_type})")
    
    print("\n=== Compact Schema Format ===")
    compact = format_schema_compact()
    print(compact)
    
    print("\n=== Enhanced Input Example ===")
    example_input = format_enhanced_input("show me flights from denver to philadelphia")
    print(example_input)
    
    print("\n=== Enhanced Target Example ===")
    example_sql = "SELECT flight_id FROM flight WHERE from_city = 'DENVER' AND to_city = 'PHILADELPHIA'"
    example_target = format_enhanced_target(example_sql)
    print(f"Target: {example_target}")
    
    print("\n=== SQL Extraction Example ===")
    mock_output = "SELECT flight_id FROM flight WHERE from_city = 'DENVER' <END> some extra text"
    extracted = extract_sql_from_output(mock_output)
    print(f"Generated: {mock_output}")
    print(f"Extracted: {extracted}")