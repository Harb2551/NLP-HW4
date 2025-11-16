#!/usr/bin/env python3
"""
Test tokenizer and model compatibility before training
"""

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import os

def test_tokenizer_model_compatibility():
    """Test that tokenizer and model work together"""
    
    print("🧪 TESTING TOKENIZER-MODEL COMPATIBILITY")
    print("="*50)
    
    # Load tokenizer
    sql_tokenizer_path = "./sql_optimized_tokenizer"
    if os.path.exists(sql_tokenizer_path):
        print("Loading SQL-optimized tokenizer...")
        tokenizer = T5TokenizerFast.from_pretrained(sql_tokenizer_path)
    else:
        print("Creating SQL-optimized tokenizer...")
        from load_data import T5Dataset
        # This will create the tokenizer
        dataset = T5Dataset('data', 'train')
        tokenizer = dataset.tokenizer
    
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Load model and resize embeddings
    print("\nLoading T5 model...")
    model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    print(f"Original model vocabulary size: {model.config.vocab_size}")
    
    # Resize embeddings
    print("Resizing model embeddings...")
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model vocabulary size: {model.config.vocab_size}")
    
    # Test tokenization and model forward pass
    print("\nTesting forward pass...")
    test_text = "Query: list flights from BOSTON to DENVER Schema: flight(flight_id, from_airport, to_airport) Answer:"
    
    # Tokenize
    inputs = tokenizer.encode(test_text, return_tensors="pt", max_length=512, truncation=True)
    print(f"Input token shape: {inputs.shape}")
    print(f"Max token ID: {inputs.max().item()}")
    print(f"Model vocab size: {model.config.vocab_size}")
    
    # Check for out-of-bounds tokens
    if inputs.max().item() >= model.config.vocab_size:
        print(f"❌ ERROR: Token ID {inputs.max().item()} >= vocab size {model.config.vocab_size}")
        return False
    
    # Test model forward pass
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=50,
                num_beams=1,
                do_sample=False
            )
        print(f"✅ Model forward pass successful!")
        print(f"Output shape: {outputs.shape}")
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False

def main():
    success = test_tokenizer_model_compatibility()
    
    if success:
        print(f"\n✅ COMPATIBILITY TEST PASSED")
        print("🚀 Ready for training!")
    else:
        print(f"\n❌ COMPATIBILITY TEST FAILED")
        print("🔧 Check tokenizer and model setup")

if __name__ == "__main__":
    main()