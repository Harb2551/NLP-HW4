#!/usr/bin/env python3
"""
Apply post-processing fixes to all predicted SQL queries
"""

import sys
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from schema_utils import extract_sql_from_output

def apply_fixes_to_predictions():
    """Apply our post-processing fixes to all predicted queries"""
    
    # Load original predicted queries
    predicted_file = "/Users/hb25/Downloads/dev.sql"
    with open(predicted_file, 'r') as f:
        original_queries = [line.strip() for line in f if line.strip()]
    
    print(f"Applying fixes to {len(original_queries)} predicted queries...")
    
    # Apply fixes to all queries
    fixed_queries = []
    stats = {
        'syntax_fixes': 0,
        'alias_fixes': 0,
        'both_fixes': 0,
        'no_fixes': 0
    }
    
    for i, original in enumerate(original_queries):
        # Apply combined fixes (syntax + alias deduplication)
        fixed = extract_sql_from_output(original + " END")
        fixed_queries.append(fixed)
        
        # Track what changed for statistics
        if fixed != original:
            # Check what type of fix was applied
            from schema_utils import fix_sql_syntax_errors, deduplicate_table_aliases
            
            syntax_fixed = fix_sql_syntax_errors(original)
            alias_fixed = deduplicate_table_aliases(original)
            
            syntax_changed = syntax_fixed != original
            alias_changed = alias_fixed != original
            
            if syntax_changed and alias_changed:
                stats['both_fixes'] += 1
            elif syntax_changed:
                stats['syntax_fixes'] += 1
            elif alias_changed:
                stats['alias_fixes'] += 1
        else:
            stats['no_fixes'] += 1
    
    # Save fixed queries to new file
    output_file = "dev_fixed.sql"
    with open(output_file, 'w') as f:
        for query in fixed_queries:
            f.write(query + '\n')
    
    print(f"\n✅ FIXES APPLIED SUCCESSFULLY")
    print(f"📊 STATISTICS:")
    print(f"  Syntax-only fixes:    {stats['syntax_fixes']:>3}")
    print(f"  Alias-only fixes:     {stats['alias_fixes']:>3}")  
    print(f"  Both fixes applied:   {stats['both_fixes']:>3}")
    print(f"  No fixes needed:      {stats['no_fixes']:>3}")
    print(f"  Total queries fixed:  {len(original_queries) - stats['no_fixes']:>3}/{len(original_queries)}")
    
    improvement_rate = ((len(original_queries) - stats['no_fixes']) / len(original_queries)) * 100
    print(f"  Improvement rate:     {improvement_rate:.1f}%")
    
    print(f"\n💾 Fixed queries saved to: {output_file}")
    print(f"📋 Next step: Re-run analysis to measure error reduction")
    
    # Show a few examples of fixes
    print(f"\n🔍 EXAMPLE FIXES:")
    count = 0
    for i, (original, fixed) in enumerate(zip(original_queries, fixed_queries)):
        if original != fixed and count < 3:
            count += 1
            print(f"\nQuery {i+1}:")
            print(f"  Before: {original[:80]}...")
            print(f"  After:  {fixed[:80]}...")
    
    return output_file

if __name__ == "__main__":
    fixed_file = apply_fixes_to_predictions()
    
    print(f"\n🚀 READY TO TEST IMPACT!")
    print(f"Run this to measure improvement:")
    print(f"python analyze_sql_errors.py {fixed_file} data/dev.sql")