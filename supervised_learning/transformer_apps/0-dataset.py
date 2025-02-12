#!/usr/bin/env python3
"""0. Dataset"""
import transformers
import tensorflow_datasets as tfds


class Dataset:
    def __init__(self):
        """Class constructor"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """tokenize"""
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        tokenizer_pt.model_max_length = 2**13
        tokenizer_en.model_max_length = 2**13
        
        return tokenizer_pt, tokenizer_en