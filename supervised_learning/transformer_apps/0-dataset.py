#!/usr/bin/env python3
"""0. Dataset"""
import transformers
import tensorflow_datasets as tfds


class Dataset:
    def __init__(self):
        """Class constructor"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        self.tokenizer_en, self.tokenizer_pt = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset"""
        pt_sentences = []
        en_sentences = []
        
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_sentences, target_vocab_size=2**13)
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_sentences, target_vocab_size=2**13)
        return tokenizer_pt, tokenizer_en
