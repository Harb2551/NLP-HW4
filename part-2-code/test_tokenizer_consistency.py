#!/usr/bin/env python3
"""
Test the tokenizer fix in training pipeline
"""

import sys
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Tokenizer
import torch
import os

def test_tokenizer_consistency():
    """Test that all tokenizers are consistent"""
    
    print("🧪 TESTING TOKENIZER CONSISTENCY")
    print("="*50)
    
    # Test 1: Load model and resize embeddings
    print("1. Loading model...")
    model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    
    # Test 2: Load SQL-optimized tokenizer
    sql_tokenizer_path = "./sql_optimized_tokenizer"
    if os.path.exists(sql_tokenizer_path):
        print("2. Loading SQL-optimized tokenizer...")
        sql_tokenizer = T5TokenizerFast.from_pretrained(sql_tokenizer_path)
        print(f"   SQL tokenizer vocab size: {len(sql_tokenizer)}")
        
        # Resize model
        print("3. Resizing model embeddings...")
        model.resize_token_embeddings(len(sql_tokenizer))
        print(f"   Model vocab size: {model.config.vocab_size}")
        
        # Test 3: Simulate evaluation tokenizer logic
        print("4. Testing evaluation tokenizer logic...")
        if os.path.exists(sql_tokenizer_path):
            eval_tokenizer = T5TokenizerFast.from_pretrained(sql_tokenizer_path)
            print("   ✅ Evaluation will use SQL-optimized tokenizer")
        else:
            eval_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
            print("   📊 Evaluation will use default tokenizer")
        
        print(f"   Eval tokenizer vocab size: {len(eval_tokenizer)}")
        
        # Test 4: Generate and decode
        print("5. Testing generation and decoding...")
        test_input = "Query: list flights Schema: flight(flight_id) Answer:"
        
        # Encode
        inputs = sql_tokenizer.encode(test_input, return_tensors="pt", max_length=100, truncation=True)
        print(f"   Input shape: {inputs.shape}, Max token ID: {inputs.max().item()}")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=50, num_beams=1)
        
        print(f"   Output shape: {outputs.shape}, Max token ID: {outputs.max().item()}")
        
        # Decode with correct tokenizer
        try:
            decoded = eval_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   ✅ Decoding successful: {decoded[:50]}...")
            return True
        except Exception as e:
            print(f"   ❌ Decoding failed: {e}")
            return False
    
    else:
        print("❌ SQL tokenizer not found")
        return False

def main():
    success = test_tokenizer_consistency()
    
    if success:
        print(f"\n✅ TOKENIZER CONSISTENCY TEST PASSED")
        print("🚀 Training should work without IndexError!")
    else:
        print(f"\n❌ TOKENIZER CONSISTENCY TEST FAILED")
        print("🔧 Check tokenizer setup")

if __name__ == "__main__":
    main()