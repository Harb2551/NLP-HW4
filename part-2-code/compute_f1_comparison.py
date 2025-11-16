#!/usr/bin/env python3
"""
Compute F1 scores for original vs fixed SQL predictions
"""

import sys
import os
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from utils import compute_records, compute_record_F1, read_queries
import time

def compute_f1_for_predictions(predicted_file, ground_truth_file, label):
    """
    Compute F1 score for a set of SQL predictions vs ground truth
    
    Args:
        predicted_file: Path to predicted SQL queries
        ground_truth_file: Path to ground truth SQL queries  
        label: Description for this evaluation
    
    Returns:
        F1 score (float)
    """
    print(f"\n{'='*60}")
    print(f"🔍 EVALUATING: {label}")
    print(f"{'='*60}")
    
    # Load queries
    predicted_queries = read_queries(predicted_file)
    gt_queries = read_queries(ground_truth_file)
    
    # Take only the same number as predictions (since we use half batches)
    num_queries = min(len(predicted_queries), len(gt_queries))
    predicted_queries = predicted_queries[:num_queries]
    gt_queries = gt_queries[:num_queries]
    
    print(f"📊 Evaluating {num_queries} query pairs")
    print(f"   Predicted from: {predicted_file}")
    print(f"   Ground truth from: {ground_truth_file}")
    
    # Compute records for predicted queries
    print(f"\n⚡ Executing predicted SQL queries...")
    start_time = time.time()
    pred_records, pred_errors = compute_records(predicted_queries)
    pred_exec_time = time.time() - start_time
    
    # Compute records for ground truth queries  
    print(f"\n⚡ Executing ground truth SQL queries...")
    start_time = time.time()
    gt_records, gt_errors = compute_records(gt_queries)
    gt_exec_time = time.time() - start_time
    
    # Count execution statistics
    pred_successful = sum(1 for error in pred_errors if not error)
    gt_successful = sum(1 for error in gt_errors if not error)
    
    print(f"\n📈 EXECUTION STATISTICS:")
    print(f"   Predicted queries executed successfully: {pred_successful}/{num_queries} ({pred_successful/num_queries*100:.1f}%)")
    print(f"   Ground truth queries executed successfully: {gt_successful}/{num_queries} ({gt_successful/num_queries*100:.1f}%)")
    print(f"   Predicted execution time: {pred_exec_time:.1f}s")
    print(f"   Ground truth execution time: {gt_exec_time:.1f}s")
    
    # Compute F1 score
    print(f"\n🎯 Computing F1 score...")
    f1_score = compute_record_F1(gt_records, pred_records)
    
    print(f"\n📊 RESULTS:")
    print(f"   F1 Score: {f1_score:.4f}")
    
    return {
        'f1_score': f1_score,
        'num_queries': num_queries,
        'pred_successful': pred_successful,
        'gt_successful': gt_successful,
        'pred_success_rate': pred_successful/num_queries,
        'gt_success_rate': gt_successful/num_queries,
        'pred_exec_time': pred_exec_time,
        'gt_exec_time': gt_exec_time
    }

def main():
    """Compare F1 scores before and after post-processing fixes"""
    
    print("🚀 F1 SCORE EVALUATION: BEFORE vs AFTER POST-PROCESSING FIXES")
    print("="*80)
    
    # File paths
    original_predictions = "/Users/hb25/Downloads/dev.sql"
    fixed_predictions = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/dev_fixed.sql"
    ground_truth = "/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code/data/dev.sql"
    
    # Check files exist
    for filepath, name in [(original_predictions, "original predictions"), 
                          (fixed_predictions, "fixed predictions"),
                          (ground_truth, "ground truth")]:
        if not os.path.exists(filepath):
            print(f"❌ ERROR: {name} file not found: {filepath}")
            return
    
    # Evaluate original predictions
    original_results = compute_f1_for_predictions(
        original_predictions, 
        ground_truth, 
        "ORIGINAL PREDICTIONS (before fixes)"
    )
    
    # Evaluate fixed predictions
    fixed_results = compute_f1_for_predictions(
        fixed_predictions, 
        ground_truth, 
        "FIXED PREDICTIONS (after post-processing)"
    )
    
    # Compare results
    print(f"\n{'='*80}")
    print(f"📈 FINAL COMPARISON: BEFORE vs AFTER")
    print(f"{'='*80}")
    
    print(f"\n🎯 F1 SCORE IMPROVEMENT:")
    print(f"   Original F1:     {original_results['f1_score']:.4f}")
    print(f"   Fixed F1:        {fixed_results['f1_score']:.4f}")
    
    f1_improvement = fixed_results['f1_score'] - original_results['f1_score']
    f1_percent_improvement = (f1_improvement / original_results['f1_score']) * 100 if original_results['f1_score'] > 0 else 0
    
    print(f"   Improvement:     +{f1_improvement:.4f} ({f1_percent_improvement:+.1f}%)")
    
    print(f"\n⚡ EXECUTION SUCCESS RATE:")
    print(f"   Original success rate: {original_results['pred_success_rate']*100:.1f}%")
    print(f"   Fixed success rate:    {fixed_results['pred_success_rate']*100:.1f}%")
    
    exec_improvement = (fixed_results['pred_success_rate'] - original_results['pred_success_rate']) * 100
    print(f"   Execution improvement: +{exec_improvement:.1f} percentage points")
    
    # Summary
    print(f"\n🏆 SUMMARY:")
    if f1_improvement > 0:
        print(f"   ✅ Post-processing fixes IMPROVED F1 score by {f1_percent_improvement:.1f}%")
    elif f1_improvement < 0:
        print(f"   ⚠️  F1 score decreased by {abs(f1_percent_improvement):.1f}% (unexpected)")
    else:
        print(f"   ➡️  F1 score unchanged (but execution rate may have improved)")
    
    if exec_improvement > 0:
        print(f"   ✅ Execution success rate improved by +{exec_improvement:.1f} percentage points")
    
    print(f"\n💡 The improved execution rate should lead to better practical performance!")

if __name__ == "__main__":
    main()