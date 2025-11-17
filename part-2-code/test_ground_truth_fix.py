"""
Removed: ground-truth fix test used during custom tokenizer debugging.
"""

raise SystemExit("test_ground_truth_fix.py removed as part of reverting custom tokenizer changes.")
#!/usr/bin/env python3
"""
Test that ground truth SQL preservation works correctly
"""

import sys
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from transformers import T5TokenizerFast
from schema_utils import extract_sql_from_output
import os

def test_ground_truth_preservation():
    """Test that ground truth SQL is not corrupted by post-processing"""
    
    print("🎯 TESTING GROUND TRUTH PRESERVATION")
    print("="*50)
    
    # Load tokenizer
    sql_tokenizer_path = "./sql_optimized_tokenizer"
    if os.path.exists(sql_tokenizer_path):
        tokenizer = T5TokenizerFast.from_pretrained(sql_tokenizer_path)
        print(f"   Using SQL-optimized tokenizer (vocab: {len(tokenizer)})")
    else:
        print("❌ SQL tokenizer not found")
        return False
    
    # Test case: Clean ground truth SQL
    print("1. Testing ground truth SQL preservation...")
    
    # Original clean SQL (what should be in ground truth)
    clean_sql = "SELECT DISTINCT flight_1.flight_id FROM flight flight_1, airport_service airport_service_1, city city_1, airport_service airport_service_2, city city_2 WHERE flight_1.departure_time BETWEEN 1200 AND 1800 AND (flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'WASHINGTON' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'BOSTON')"
    
    # Simulate what happens in the pipeline
    # 1. Tokenize and decode (simulate model target)
    tokens = tokenizer.encode(clean_sql + " <END>", return_tensors="pt")
    decoded_raw = tokenizer.decode(tokens[0], skip_special_tokens=True).strip()
    
    print(f"   Original SQL: {clean_sql[:100]}...")
    print(f"   After tokenize/decode: {decoded_raw[:100]}...")
    
    # 2. Apply our new ground truth processing (just remove <END>)
    processed_gt = decoded_raw.replace('<END>', '').strip()
    print(f"   After GT processing: {processed_gt[:100]}...")
    
    # 3. Compare with old method (extract_sql_from_output)
    old_method = extract_sql_from_output(decoded_raw)
    print(f"   Old method (corrupted): {old_method[:100]}...")
    
    # Check preservation
    spaces_preserved = " FROM " in processed_gt and " WHERE " in processed_gt
    not_corrupted = "FROMflight" not in processed_gt and "WHEREflight" not in processed_gt
    
    if spaces_preserved and not_corrupted:
        print("✅ Ground truth SQL properly preserved with spaces")
        return True
    else:
        print("❌ Ground truth SQL still corrupted")
        return False

def main():
    success = test_ground_truth_preservation()
    
    if success:
        print(f"\n✅ GROUND TRUTH PRESERVATION TEST PASSED")
        print("🎯 Ground truth SQL will maintain proper formatting!")
    else:
        print(f"\n❌ GROUND TRUTH PRESERVATION TEST FAILED")
        print("🔧 Need to fix ground truth processing")

if __name__ == "__main__":
    main()