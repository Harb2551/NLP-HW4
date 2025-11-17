"""
Removed: fix validation tests tied to custom tokenizer outputs.
"""

raise SystemExit("test_fixes.py removed as part of reverting custom tokenizer changes.")
#!/usr/bin/env python3
"""
Test SQL post-processing fixes on actual predicted queries
"""

import sys
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from schema_utils import fix_sql_syntax_errors, deduplicate_table_aliases, extract_sql_from_output

def test_fixes_on_predictions():
    """Test our fixes on the actual predicted queries"""
    
    # Load predicted queries
    predicted_file = "/Users/hb25/Downloads/dev.sql"
    with open(predicted_file, 'r') as f:
        predicted_queries = [line.strip() for line in f if line.strip()]
    
    print(f"Testing fixes on {len(predicted_queries)} predicted queries...")
    
    # Count improvements
    syntax_fixes = 0
    alias_fixes = 0
    both_fixes = 0
    
    # Test first 10 queries
    for i in range(min(10, len(predicted_queries))):
        original = predicted_queries[i]
        
        # Apply individual fixes to track what changed
        syntax_fixed = fix_sql_syntax_errors(original)
        alias_fixed = deduplicate_table_aliases(original)
        both_fixed = extract_sql_from_output(original + " END")
        
        syntax_changed = syntax_fixed != original
        alias_changed = alias_fixed != original
        
        if syntax_changed:
            syntax_fixes += 1
        if alias_changed:
            alias_fixes += 1
        if syntax_changed and alias_changed:
            both_fixes += 1
            
        # Show examples of fixes
        if syntax_changed or alias_changed:
            print(f"\n=== Query {i+1} Fixes ===")
            print(f"Original:  {original[:100]}...")
            if syntax_changed:
                print(f"Syntax:    {syntax_fixed[:100]}...")
            if alias_changed:
                print(f"Aliases:   {alias_fixed[:100]}...")
            print(f"Combined:  {both_fixed[:100]}...")
    
    print(f"\n=== SUMMARY (first 10 queries) ===")
    print(f"Syntax fixes applied: {syntax_fixes}/10")
    print(f"Alias fixes applied: {alias_fixes}/10")
    print(f"Both fixes applied: {both_fixes}/10")
    
    # Test a few specific error patterns from our analysis
    print(f"\n=== SPECIFIC ERROR PATTERN TESTS ===")
    
    # Test malformed AND conditions
    test_and_error = "SELECT flight_1.flight_id FROM flight flight_1 WHERE city_1.city_name = 'DENVER' AND( flight_1.arrival_time = 1700 )"
    fixed_and = fix_sql_syntax_errors(test_and_error)
    print(f"AND( fix:")
    print(f"  Before: {test_and_error}")
    print(f"  After:  {fixed_and}")
    
    # Test duplicate aliases
    test_dup_aliases = "SELECT DISTINCT flight_1.flight_id FROM flight flight_1, city city_1, airport_service airport_service_2, city city_1, airport_service airport_service_2"
    fixed_dup = deduplicate_table_aliases(test_dup_aliases)
    print(f"\nDuplicate alias fix:")
    print(f"  Before: {test_dup_aliases}")
    print(f"  After:  {fixed_dup}")

if __name__ == "__main__":
    test_fixes_on_predictions()