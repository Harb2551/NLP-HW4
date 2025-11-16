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

from schema_utils import format_enhanced_input, format_enhanced_target
from sql_preprocessing import preprocess_sql_for_tokenization, preprocess_nl_for_tokenization

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Dataset class for performing data processing for the T5 model.
        '''
        self.split = split
        self.data_folder = data_folder
        
        # Initialize SQL-optimized T5 tokenizer (create if needed)
        sql_tokenizer_path = "./sql_optimized_tokenizer"
        if os.path.exists(sql_tokenizer_path):
            print("🚀 Using existing SQL-optimized tokenizer")
            self.tokenizer = T5TokenizerFast.from_pretrained(sql_tokenizer_path)
        else:
            print("� Creating SQL-optimized tokenizer...")
            self.tokenizer = self._create_sql_tokenizer(sql_tokenizer_path)
    
    def _create_sql_tokenizer(self, save_path):
        """Create and save SQL-optimized tokenizer"""
        
        # Start with base tokenizer
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # SQL-specific vocabulary
        sql_vocab = [
            'SELECT_DISTINCT', 'SELECT', 'FROM', 'WHERE', 'AND_', 'OR_',
            'JOIN', 'INNER_JOIN', 'LEFT_JOIN', 'ON', 'GROUP_BY', 'ORDER_BY',
            'HAVING', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
            '_=_', '_<_', '_>_', '_<=_', '_>=_', '_!=_',
            'flight_1', 'flight_2', 'airport_1', 'airport_2', 'city_1', 'city_2',
            'airline_1', 'ground_service_1', 'aircraft_1', 'fare_1', 'equipment_sequence_1',
            'flight_fare_1', 'food_service_1', 'airport_service_1',
            'flight_id', 'airport_code', 'city_code', 'departure_time', 'arrival_time',
            'airline_code', 'aircraft_code', 'flight_number', 'from_airport', 'to_airport',
            'city_name', 'state_code', 'transport_type', 'meal_code',
            'flight_1.flight_id', 'flight_1.from_airport', 'flight_1.to_airport',
            'flight_1.departure_time', 'flight_1.arrival_time', 'flight_1.airline_code',
            'airport_1.airport_code', 'city_1.city_code', 'city_1.city_name',
            "'BOS'", "'DEN'", "'ATL'", "'LAX'", "'JFK'", "'LGA'", "'SFO'", "'ORD'",
            "'MIA'", "'SEA'", "'DFW'", "'PHX'", "'LAS'", "'CLT'", "'MKE'", "'PIT'",
            "'BOSTON'", "'DENVER'", "'ATLANTA'", "'DALLAS'", "'NEW_YORK'", "'PHILADELPHIA'",
            "'UA'", "'AA'", "'DL'", "'WN'", "'US'", "'NW'", "'CO'", "'AS'",
            '800', '900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800',
            "WHERE_1=1", "AND_1=1",
        ]
        
        # Add new tokens
        existing_vocab = tokenizer.get_vocab()
        new_tokens = [token for token in sql_vocab if token not in existing_vocab]
        
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
            print(f"Added {len(new_tokens)} SQL-specific tokens")
        
        # Save tokenizer
        tokenizer.save_pretrained(save_path)
        print(f"💾 Saved SQL tokenizer to {save_path}")
        
        return tokenizer
        
        # Process and load data
        self.data = self.process_data(data_folder, split, self.tokenizer)
        
        print(f"Loaded {len(self.data)} examples for {split} split")

    def process_data(self, data_folder, split, tokenizer):
        '''
        Load and process the natural language and SQL data
        '''
        # Load natural language queries - use preprocessed for training
        if split == "train":
            preprocessed_nl_file = os.path.join(data_folder, "train_preprocessed.nl")
            if os.path.exists(preprocessed_nl_file):
                print("📊 Using preprocessed training data for T5")
                nl_file = preprocessed_nl_file
            else:
                print("📊 Using original training data for T5")
                nl_file = os.path.join(data_folder, f"{split}.nl")
        else:
            nl_file = os.path.join(data_folder, f"{split}.nl")
            
        with open(nl_file, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines()]
        
        data = []
        
        if split == "test":
            # Test set: only has natural language queries
            for i, nl_query in enumerate(nl_queries):
                # Preprocess NL query for better tokenization
                processed_nl = preprocess_nl_for_tokenization(nl_query)
                
                # Create enhanced input with schema information and Answer: pattern
                enhanced_input = format_enhanced_input(processed_nl)
                
                # Tokenize enhanced input
                encoder_input = tokenizer.encode(
                    enhanced_input,
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
            if split == "train":
                preprocessed_sql_file = os.path.join(data_folder, "train_preprocessed.sql")
                if os.path.exists(preprocessed_sql_file):
                    sql_file = preprocessed_sql_file
                else:
                    sql_file = os.path.join(data_folder, f"{split}.sql")
            else:
                sql_file = os.path.join(data_folder, f"{split}.sql")
                
            with open(sql_file, 'r') as f:
                sql_queries = [line.strip() for line in f.readlines()]
            
            assert len(nl_queries) == len(sql_queries), f"Mismatch in {split}: {len(nl_queries)} vs {len(sql_queries)}"
            
            for i, (nl_query, sql_query) in enumerate(zip(nl_queries, sql_queries)):
                # Preprocess queries for better tokenization
                processed_nl = preprocess_nl_for_tokenization(nl_query)
                processed_sql = preprocess_sql_for_tokenization(sql_query)
                
                # Create enhanced input with schema information and Answer: pattern
                enhanced_input = format_enhanced_input(processed_nl)
                
                # Tokenize enhanced input
                encoder_input = tokenizer.encode(
                    enhanced_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).squeeze(0)
                
                # Format and tokenize SQL target
                formatted_target = format_enhanced_target(processed_sql)
                decoder_target = tokenizer.encode(
                    formatted_target,
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
    
    # Use preprocessed training data if available, otherwise fall back to original
    train_nl_file = os.path.join(data_folder, 'train_preprocessed.nl')
    train_sql_file = os.path.join(data_folder, 'train_preprocessed.sql')
    
    if os.path.exists(train_nl_file) and os.path.exists(train_sql_file):
        print("📊 Using preprocessed training data")
        train_nl = load_lines(train_nl_file)
        train_sql = load_lines(train_sql_file)
    else:
        print("📊 Using original training data")
        train_nl = load_lines(os.path.join(data_folder, 'train.nl'))
        train_sql = load_lines(os.path.join(data_folder, 'train.sql'))
    
    dev_nl = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_sql = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_nl = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_nl, train_sql, dev_nl, dev_sql, test_nl