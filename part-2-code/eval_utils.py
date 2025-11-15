"""
Evaluation utilities for T5 text-to-SQL model.

This module contains functions for evaluating T5 model performance,
including SQL generation and F1 score computation.
"""

import torch
from utils import compute_record_F1, compute_records, read_queries
from schema_utils import extract_sql_from_output


def rerank_candidates_by_execution(candidates, target_sql=None, tokenizer=None):
    """
    Rerank SQL candidates by execution success only.
    
    At evaluation time, we should only use execution success to select candidates,
    not target overlap (which would be cheating since we don't have ground truth
    during actual inference).
    
    Args:
        candidates: List of SQL candidate strings
        target_sql: Target SQL string (unused, kept for interface consistency)
        tokenizer: Tokenizer (unused but kept for interface consistency)
    
    Returns:
        Best SQL candidate string (first one that executes successfully)
    """
    if not candidates:
        return ""
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Use optimized batch execution but suppress progress bars
    # This maintains the threading performance of compute_records
    import os
    from contextlib import redirect_stderr
    from io import StringIO
    
    # Temporarily suppress tqdm progress bars by redirecting stderr
    f = StringIO()
    with redirect_stderr(f):
        candidate_records, candidate_errors = compute_records(candidates)
    
    # Return the first candidate that executes successfully
    for candidate, error in zip(candidates, candidate_errors):
        if not error:
            return candidate
    
    # If all candidates failed, return the first one
    return candidates[0]

def eval_epoch(model, dataloader, tokenizer, device, generation_max_length=256, 
               num_beams=1, num_candidates=1, rerank_by_execution=False, return_predictions=False):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model: The T5 model to evaluate
        dataloader: DataLoader for the evaluation data
        tokenizer: T5 tokenizer
        device: Device to run evaluation on
        generation_max_length: Maximum length for generated SQL queries
        num_beams: Number of beams for beam search (1 = greedy decoding)
        num_candidates: Number of candidates to generate per input (for reranking)
        rerank_by_execution: If True, generate multiple candidates and rerank by execution success
        return_predictions: If True, return generated predictions alongside F1 score
    
    Returns:
        If return_predictions is False: float (F1 score)
        If return_predictions is True: tuple (F1 score, list of predictions)
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    # Use only half the batches for faster evaluation
    max_batches = len(dataloader) // 2
    print(f"Evaluating on {max_batches}/{len(dataloader)} batches (half for speed)...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Stop after processing half the batches
            if batch_idx >= max_batches:
                break
            # Handle different batch formats (train vs test)
            if len(batch) == 5:  # Train/dev format
                encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs = batch
            else:  # Test format
                encoder_ids, encoder_mask, initial_decoder_inputs = batch
                decoder_targets = None
            
            # Move to device
            encoder_ids = encoder_ids.to(device)
            encoder_mask = encoder_mask.to(device)
            initial_decoder_inputs = initial_decoder_inputs.to(device)
            
            # Generate SQL queries
            if rerank_by_execution and num_candidates > 1:
                # Generate multiple candidates for reranking
                generated_ids = model.generate(
                    input_ids=encoder_ids,
                    attention_mask=encoder_mask,
                    decoder_start_token_id=initial_decoder_inputs[:, 0],  # BOS token
                    max_length=generation_max_length,
                    num_beams=max(num_beams, num_candidates),
                    num_return_sequences=num_candidates,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,  # Use deterministic beam search
                )
            else:
                # Standard generation
                generated_ids = model.generate(
                    input_ids=encoder_ids,
                    attention_mask=encoder_mask,
                    decoder_start_token_id=initial_decoder_inputs[:, 0],  # BOS token
                    max_length=generation_max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Process generated sequences
            batch_size = encoder_ids.shape[0]
            sequences_per_input = num_candidates if (rerank_by_execution and num_candidates > 1) else 1
            
            for i in range(batch_size):
                if rerank_by_execution and num_candidates > 1:
                    # Extract candidates for this input
                    start_idx = i * sequences_per_input
                    end_idx = start_idx + sequences_per_input
                    candidate_ids = generated_ids[start_idx:end_idx]
                    
                    # Decode all candidates
                    candidates = []
                    for cand_id in candidate_ids:
                        candidate_raw = tokenizer.decode(cand_id, skip_special_tokens=True).strip()
                        candidate_sql = extract_sql_from_output(candidate_raw)
                        candidates.append(candidate_sql)
                    
                    # Rerank candidates by execution success
                    best_sql = rerank_candidates_by_execution(candidates, decoder_targets[i] if decoder_targets is not None else None, tokenizer)
                    all_predictions.append(best_sql)
                else:
                    # Standard single prediction
                    generated_raw = tokenizer.decode(
                        generated_ids[i], 
                        skip_special_tokens=True
                    ).strip()
                    generated_sql = extract_sql_from_output(generated_raw)
                    all_predictions.append(generated_sql)
                
                # Get target SQL if available (for train/dev)
                if decoder_targets is not None:
                    target_raw = tokenizer.decode(
                        decoder_targets[i], 
                        skip_special_tokens=True
                    ).strip()
                    target_sql = extract_sql_from_output(target_raw)
                    all_targets.append(target_sql)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{max_batches} batches")
    
    # Compute F1 score if we have targets (requires executing SQL queries on database)
    if all_targets:
        print("Executing SQL queries on database to compute F1 score...")
        
        # Execute generated SQL queries to get records
        print("  Executing predicted SQL queries...")
        pred_records, pred_errors = compute_records(all_predictions)
        
        # Execute ground truth SQL queries to get records  
        print("  Executing ground truth SQL queries...")
        # Clean ground truth SQL by removing <END> tokens before execution
        clean_targets = [extract_sql_from_output(target) for target in all_targets]
        gt_records, gt_errors = compute_records(clean_targets)
        
        # Compute F1 score based on database records (not SQL strings)
        f1_score = compute_record_F1(gt_records, pred_records)
        print(f"Record-based F1 Score: {f1_score:.4f}")
        
        # Report any SQL execution errors
        pred_error_count = sum(1 for err in pred_errors if err)
        gt_error_count = sum(1 for err in gt_errors if err)
        if pred_error_count > 0:
            print(f"  ⚠️  {pred_error_count}/{len(pred_errors)} predicted queries had execution errors")
        if gt_error_count > 0:
            print(f"  ⚠️  {gt_error_count}/{len(gt_errors)} ground truth queries had execution errors")
        
        # Print some example predictions
        print("\nSample predictions:")
        for i in range(min(3, len(all_predictions))):
            print(f"  Prediction: {all_predictions[i]}")
            print(f"  Target:     {all_targets[i]}")
            if pred_errors[i]:
                print(f"  Pred Error: {pred_errors[i]}")
            if gt_errors[i]:
                print(f"  GT Error:   {gt_errors[i]}")
            print(f"  Pred Records: {len(pred_records[i])} records")
            print(f"  GT Records:   {len(gt_records[i])} records")
            print()
    else:
        f1_score = None
        print("No targets available - generating predictions for test set")
    
    model.train()  # Reset to training mode
    
    if return_predictions:
        return f1_score, all_predictions
    else:
        return f1_score


def save_predictions_to_file(predictions, filename):
    """
    Save predictions to a file, one per line.
    
    Args:
        predictions: List of prediction strings
        filename: Output filename
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    
    print(f"Saved {len(predictions)} predictions to {filename}")


def evaluate_and_save(model, dataloader, tokenizer, device, output_file, 
                     generation_max_length=256, num_beams=1, num_candidates=1, rerank_by_execution=False):
    """
    Evaluate model and save predictions to file.
    
    Args:
        model: The T5 model to evaluate
        dataloader: DataLoader for the evaluation data
        tokenizer: T5 tokenizer
        device: Device to run evaluation on
        output_file: File to save predictions to
        generation_max_length: Maximum length for generated SQL queries
        num_beams: Number of beams for beam search
        num_candidates: Number of candidates to generate per input (for reranking)
        rerank_by_execution: If True, generate multiple candidates and rerank by execution success
    
    Returns:
        F1 score (None if no targets available)
    """
    f1_score, predictions = eval_epoch(
        model=model,
        dataloader=dataloader, 
        tokenizer=tokenizer,
        device=device,
        generation_max_length=generation_max_length,
        num_beams=num_beams,
        num_candidates=num_candidates,
        rerank_by_execution=rerank_by_execution,
        return_predictions=True
    )
    
    save_predictions_to_file(predictions, output_file)
    
    return f1_score