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
        Formatted target string with END token
    """
    return sql_query.strip() + " END"

def extract_sql_from_output(generated_output: str) -> str:
    """
    Extract SQL query from generated output by finding END token.
    Apply post-processing to fix common errors.
    
    Args:
        generated_output: The raw generated text from the model
    
    Returns:
        Extracted and cleaned SQL query without the END token
    """
    output = generated_output.strip()
    
    # Look for END token
    if " END" in output:
        sql = output.split(" END")[0].strip()
    else:
        # Fallback: use the entire output if no END token found
        sql = output
    
    # Apply post-processing fixes
    sql = fix_sql_syntax_errors(sql)
    sql = deduplicate_table_aliases(sql)
    
    return sql

def fix_sql_syntax_errors(sql: str) -> str:
    """
    Fix common SQL syntax errors in generated queries.
    
    Args:
        sql: Raw SQL query string
        
    Returns:
        SQL with syntax errors fixed
    """
    import re
    
    # Fix malformed AND/OR conditions: "AND(" -> "AND ("
    sql = re.sub(r'\bAND\(', 'AND (', sql)
    sql = re.sub(r'\bOR\(', 'OR (', sql)
    
    # Fix missing spaces around operators
    sql = re.sub(r'(\w)=(\w)', r'\1 = \2', sql)
    sql = re.sub(r'(\w)<(\w)', r'\1 < \2', sql)
    sql = re.sub(r'(\w)>(\w)', r'\1 > \2', sql)
    
    # Fix missing comparison operators for time/numeric conditions
    # Pattern: column_name followed by number without operator
    # arrival_time 900 -> arrival_time < 900 (assume < for arrival times)
    sql = re.sub(r'(\w+\.(?:arrival_time|departure_time))\s+(\d+)(?!\d)(?!\s*[=<>])', 
                 r'\1 < \2', sql)
    
    # capacity 100 -> capacity >= 100 (assume >= for capacity)
    sql = re.sub(r'(\w+\.capacity)\s+(\d+)(?!\d)(?!\s*[=<>])', 
                 r'\1 >= \2', sql)
    
    # General numeric comparisons without operators - default to =
    sql = re.sub(r'(\w+\.\w+)\s+(\d+)(?!\d)(?!\s*[=<>])', 
                 r'\1 = \2', sql)
    
    return sql

def deduplicate_table_aliases(sql: str) -> str:
    """
    Remove duplicate table aliases from FROM clause.
    
    Args:
        sql: SQL query string
        
    Returns:
        SQL with deduplicated table aliases
    """
    import re
    
    # Find FROM clause
    from_match = re.search(r'FROM\s+(.+?)(?:\s+WHERE|$)', sql, re.IGNORECASE | re.DOTALL)
    if not from_match:
        return sql
    
    from_clause = from_match.group(1).strip()
    
    # Parse table alias pairs: "table_name alias_name"
    table_alias_pairs = re.findall(r'(\w+)\s+(\w+_\d+)', from_clause)
    
    if not table_alias_pairs:
        return sql  # No aliases found
    
    # Track seen aliases and build unique list
    seen_aliases = set()
    unique_pairs = []
    
    for table, alias in table_alias_pairs:
        if alias not in seen_aliases:
            unique_pairs.append(f"{table} {alias}")
            seen_aliases.add(alias)
        # Skip duplicates
    
    # Rebuild FROM clause with comma separation
    new_from_clause = ", ".join(unique_pairs)
    
    # Replace the FROM clause in the original query
    if "WHERE" in sql.upper():
        replacement = f"FROM {new_from_clause} WHERE"
        sql = re.sub(r'FROM\s+.+?\s+WHERE', replacement, sql, flags=re.IGNORECASE | re.DOTALL)
    else:
        replacement = f"FROM {new_from_clause}"
        sql = re.sub(r'FROM\s+.+?$', replacement, sql, flags=re.IGNORECASE | re.DOTALL)
    
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
    mock_output = "SELECT flight_id FROM flight WHERE from_city = 'DENVER' END some extra text"
    extracted = extract_sql_from_output(mock_output)
    print(f"Generated: {mock_output}")
    print(f"Extracted: {extracted}")
    
    print("\n=== Error Fixing Examples ===")
    
    # Test syntax error fixing
    syntax_error_sql = "SELECT flight_id FROM flight WHERE city='DENVER' AND( arrival_time<900 )"
    fixed_syntax = fix_sql_syntax_errors(syntax_error_sql)
    print(f"Syntax Fix:")
    print(f"  Before: {syntax_error_sql}")
    print(f"  After:  {fixed_syntax}")
    
    # Test missing operator fixing
    missing_op_sql = "SELECT flight_1.flight_id FROM flight flight_1 WHERE flight_1.arrival_time 900 AND flight_1.capacity 200"
    fixed_operators = fix_sql_syntax_errors(missing_op_sql)
    print(f"\nMissing Operator Fix:")
    print(f"  Before: {missing_op_sql}")
    print(f"  After:  {fixed_operators}")
    
    # Test duplicate alias fixing
    dup_alias_sql = "SELECT DISTINCT flight_1.flight_id FROM flight flight_1, city city_1, airport_service airport_service_2, city city_1, airport_service airport_service_2 WHERE flight_1.from_airport = city_1.city_code"
    fixed_aliases = deduplicate_table_aliases(dup_alias_sql)
    print(f"\nAlias Deduplication:")
    print(f"  Before: {dup_alias_sql}")
    print(f"  After:  {fixed_aliases}")
    
    # Test combined fixing
    combined_errors = "SELECT flight_1.flight_id FROM flight flight_1, city city_1, city city_1 WHERE city_1.city_name='DENVER' AND( flight_1.arrival_time<900 ) END"
    fixed_combined = extract_sql_from_output(combined_errors)
    print(f"\nCombined Fixes:")
    print(f"  Before: {combined_errors}")
    print(f"  After:  {fixed_combined}")