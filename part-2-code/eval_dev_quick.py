"""
Removed: quick dev evaluation script specific to custom tokenizer experiments.
"""

raise SystemExit("eval_dev_quick.py removed as part of reverting custom tokenizer changes.")
#!/usr/bin/env python3
"""
Quick dev evaluation to sanity-check GT formatting and prediction normalization
"""

import os
import types
from types import SimpleNamespace

import torch
from transformers import T5TokenizerFast

from load_data import get_dataloader
from eval_utils import eval_epoch
from t5_utils import load_model_from_checkpoint, DEVICE


def main():
    # Load tokenizer (SQL-optimized if present)
    sql_tokenizer_path = "./sql_optimized_tokenizer"
    if os.path.exists(sql_tokenizer_path):
        tokenizer = T5TokenizerFast.from_pretrained(sql_tokenizer_path)
        print(f"Using SQL-optimized tokenizer (vocab={len(tokenizer)})")
    else:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
        print(f"Using default T5 tokenizer (vocab={len(tokenizer)})")

    # Load dev dataloader (small batch)
    dev_loader = get_dataloader(batch_size=4, split="dev")

    # Prepare args for loading checkpoint
    args = SimpleNamespace(
        finetune=True,
        experiment_name='ft_wandb_sanity',
        checkpoint_dir=os.path.join('runs', 'ft_experiments', 'ft_wandb_sanity', 'checkpoints')
    )

    # Load best checkpoint
    model = load_model_from_checkpoint(args, best=True)
    model.to(DEVICE)
    model.eval()

    # Evaluate a small slice using our eval utils (which now preserves GT and normalizes preds)
    f1, preds = eval_epoch(
        model=model,
        dataloader=dev_loader,
        tokenizer=tokenizer,
        device=DEVICE,
        generation_max_length=256,
        num_beams=1,
        num_candidates=1,
        rerank_by_execution=False,
        return_predictions=True,
    )

    print(f"\nDev quick check — Record-based F1: {f1:.4f}")
    for i, p in enumerate(preds[:3]):
        print(f"Pred {i+1}: {p}")


if __name__ == "__main__":
    main()
