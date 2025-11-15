import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Dataset class for performing data processing for the T5 model.
        '''
        self.split = split
        self.data_folder = data_folder
        
        # Initialize T5 tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Process and load data
        self.data = self.process_data(data_folder, split, self.tokenizer)
        
        print(f"Loaded {len(self.data)} examples for {split} split")

    def process_data(self, data_folder, split, tokenizer):
        '''
        Load and process the natural language and SQL data
        '''
        # Load natural language queries
        nl_file = os.path.join(data_folder, f"{split}.nl")
        with open(nl_file, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines()]
        
        data = []
        
        if split == "test":
            # Test set: only has natural language queries
            for i, nl_query in enumerate(nl_queries):
                # Tokenize natural language input
                encoder_input = tokenizer.encode(
                    f"translate English to SQL: {nl_query}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).squeeze(0)
                
                data.append({
                    'encoder_input': encoder_input,
                    'nl_query': nl_query,
                    'idx': i
                })
        else:
            # Train/dev sets: have both NL queries and SQL targets
            sql_file = os.path.join(data_folder, f"{split}.sql")
            with open(sql_file, 'r') as f:
                sql_queries = [line.strip() for line in f.readlines()]
            
            assert len(nl_queries) == len(sql_queries), f"Mismatch in {split}: {len(nl_queries)} vs {len(sql_queries)}"
            
            for i, (nl_query, sql_query) in enumerate(zip(nl_queries, sql_queries)):
                # Tokenize natural language input with T5 prefix
                encoder_input = tokenizer.encode(
                    f"translate English to SQL: {nl_query}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).squeeze(0)
                
                # Tokenize SQL target
                decoder_target = tokenizer.encode(
                    sql_query,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).squeeze(0)
                
                # Create decoder input (shift right with BOS token)
                decoder_input = torch.cat([
                    torch.tensor([tokenizer.pad_token_id]),  # BOS token
                    decoder_target[:-1]  # All but last token
                ])
                
                data.append({
                    'encoder_input': encoder_input,
                    'decoder_input': decoder_input,
                    'decoder_target': decoder_target,
                    'nl_query': nl_query,
                    'sql_query': sql_query,
                    'idx': i
                })
        
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.
    '''
    # Extract components from batch
    encoder_inputs = [item['encoder_input'] for item in batch]
    decoder_inputs = [item['decoder_input'] for item in batch]
    decoder_targets = [item['decoder_target'] for item in batch]
    
    # Pad sequences to the same length in the batch
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_input_ids = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_target_ids = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention masks (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder inputs (for generation during evaluation)
    initial_decoder_inputs = torch.tensor([PAD_IDX] * len(batch)).unsqueeze(1)  # BOS tokens
    
    return encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.
    '''
    # Extract encoder inputs from batch
    encoder_inputs = [item['encoder_input'] for item in batch]
    
    # Pad sequences to the same length in the batch
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention masks (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder inputs (BOS tokens for generation)
    initial_decoder_inputs = torch.tensor([PAD_IDX] * len(batch)).unsqueeze(1)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Load data for prompting approach (not T5)
    # This function is used by the prompting.py script
    train_nl = load_lines(os.path.join(data_folder, 'train.nl'))
    train_sql = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_nl = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_sql = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_nl = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_nl, train_sql, dev_nl, dev_sql, test_nl