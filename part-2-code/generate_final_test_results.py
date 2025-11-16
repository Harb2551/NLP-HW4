#!/usr/bin/env python3
"""
Apply post-processing fixes to final test results and generate clean outputs
"""

import sys
import os
import pickle
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from schema_utils import extract_sql_from_output
from utils import compute_records, read_queries, save_queries_and_records

def process_test_results():
    """
    Apply post-processing fixes to test results and generate final outputs
    """
    
    # File paths - save in same directory as original test file
    test_results_dir = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/runs/finetune_fast_eval_enhanced_101ep/results"
    original_test_file = os.path.join(test_results_dir, "test.sql")
    output_sql_file = os.path.join(test_results_dir, "test_post_processed.sql")
    output_pkl_file = os.path.join(test_results_dir, "test_post_processed.pkl")
    
    print("🚀 POST-PROCESSING FINAL TEST RESULTS")
    print("="*60)
    
    # Check if original test file exists
    if not os.path.exists(original_test_file):
        print(f"❌ ERROR: Original test file not found: {original_test_file}")
        return
    
    # Load original test queries
    print(f"📂 Loading original test queries from:")
    print(f"   {original_test_file}")
    
    original_queries = read_queries(original_test_file)
    print(f"   Loaded {len(original_queries)} test queries")
    
    # Apply post-processing fixes
    print(f"\n🔧 Applying post-processing fixes...")
    fixed_queries = []
    stats = {
        'syntax_fixes': 0,
        'alias_fixes': 0, 
        'both_fixes': 0,
        'no_fixes': 0
    }
    
    for i, original in enumerate(original_queries):
        # Apply combined fixes (same as we did for dev set)
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
    
    # Save fixed SQL queries and generate records using the same function as test_inference
    print(f"\n💾 Saving post-processed SQL queries and generating records...")
    try:
        # Use the exact same function that test_inference uses
        save_queries_and_records(fixed_queries, output_sql_file, output_pkl_file)
        
        print(f"   SQL saved to: {output_sql_file}")
        print(f"   Records saved to: {output_pkl_file}")
        
        # Count execution success for statistics
        records, errors = compute_records(fixed_queries)
        successful_executions = sum(1 for error in errors if not error)
        print(f"   Successfully executed: {successful_executions}/{len(fixed_queries)} ({successful_executions/len(fixed_queries)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error saving queries and records: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return
    
    # Print summary statistics
    print(f"\n📊 POST-PROCESSING STATISTICS:")
    print(f"   Original queries:       {len(original_queries)}")
    print(f"   Syntax-only fixes:      {stats['syntax_fixes']}")
    print(f"   Alias-only fixes:       {stats['alias_fixes']}")
    print(f"   Both fixes applied:     {stats['both_fixes']}")
    print(f"   No fixes needed:        {stats['no_fixes']}")
    
    total_fixes = stats['syntax_fixes'] + stats['alias_fixes'] + stats['both_fixes']
    print(f"   Total queries fixed:    {total_fixes}/{len(original_queries)} ({total_fixes/len(original_queries)*100:.1f}%)")
    print(f"   Execution success rate: {successful_executions}/{len(fixed_queries)} ({successful_executions/len(fixed_queries)*100:.1f}%)")
    
    # Show a few examples of fixes
    print(f"\n🔍 EXAMPLE FIXES:")
    count = 0
    for i, (original, fixed) in enumerate(zip(original_queries, fixed_queries)):
        if original != fixed and count < 3:
            count += 1
            print(f"\nTest Query {i+1}:")
            print(f"  Before: {original[:80]}...")
            print(f"  After:  {fixed[:80]}...")
    
    print(f"\n✅ SUCCESS! Final test outputs ready:")
    print(f"   📄 SQL queries: {output_sql_file}")
    print(f"   📦 Records:     {output_pkl_file}")
    
    # Verify pickle file format for autograder compatibility
    print(f"\n🔍 Verifying pickle file format...")
    try:
        with open(output_pkl_file, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Should be a tuple of (records, errors)
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            records_part, errors_part = loaded_data
            print(f"   ✅ Correct format: tuple with (records, errors)")
            print(f"   Records count: {len(records_part)}")
            print(f"   Errors count: {len(errors_part)}")
            if records_part:
                print(f"   First record type: {type(records_part[0])}")
                print(f"   First record sample: {str(records_part[0])[:100]}...")
            print(f"   ✅ Pickle file format matches save_queries_and_records")
        else:
            print(f"   ⚠️ Unexpected format: {type(loaded_data)}")
            
    except Exception as e:
        print(f"   ⚠️ Warning: Could not verify pickle file: {e}")
    
    print(f"\n🎯 These files are ready for Gradescope submission!")

def main():
    process_test_results()

if __name__ == "__main__":
    main()