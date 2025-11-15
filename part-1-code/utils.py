import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Intelligent Synonym Replacement Transformation
    # This transformation replaces words with their synonyms using WordNet, simulating
    # how different users might express the same sentiment with varied vocabulary.
    
    text = example["text"]
    tokens = word_tokenize(text)
    transformed_tokens = []
    
    # Probability of replacing each word (increased to 60% for more aggressive transformation)
    replacement_prob = 0.6
    
    # Reduced avoid_words list - only keep the most essential function words
    avoid_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                   'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                   'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                   'should', 'may', 'might', 'can', 'must'}
    
    for token in tokens:
        # More aggressive: reduce minimum word length and allow more transformations
        if (not token.isalpha() or 
            token.lower() in avoid_words or 
            len(token) < 2 or  # Changed from 3 to 2
            random.random() > replacement_prob):
            transformed_tokens.append(token)
            continue
        
        # Get synonyms from WordNet - more aggressive synonym selection
        synonyms = set()
        for syn in wordnet.synsets(token.lower()):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                # Allow more diverse synonyms, including slightly different forms
                if ' ' not in synonym and synonym.lower() != token.lower() and len(synonym) >= 2:
                    synonyms.add(synonym)
        
        # If no direct synonyms, try getting antonyms and other related words for more aggressive transformation
        if not synonyms:
            for syn in wordnet.synsets(token.lower()):
                # Get hypernyms (more general terms) and hyponyms (more specific terms)
                for hypernym in syn.hypernyms():
                    for lemma in hypernym.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if ' ' not in synonym and synonym.lower() != token.lower() and len(synonym) >= 2:
                            synonyms.add(synonym)
                
                for hyponym in syn.hyponyms():
                    for lemma in hyponym.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if ' ' not in synonym and synonym.lower() != token.lower() and len(synonym) >= 2:
                            synonyms.add(synonym)
        
        if synonyms:
            # Choose a random synonym, preserving original capitalization
            chosen_synonym = random.choice(list(synonyms))
            if token.isupper():
                chosen_synonym = chosen_synonym.upper()
            elif token.istitle():
                chosen_synonym = chosen_synonym.capitalize()
            transformed_tokens.append(chosen_synonym)
        else:
            # No suitable synonyms found, keep original
            transformed_tokens.append(token)
    
    # Reconstruct the text
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
