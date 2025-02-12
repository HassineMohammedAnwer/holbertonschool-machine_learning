#!/usr/bin/env python3
"""1. Dataset"""
import transformers
import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    """class dataset"""
    def __init__(self):
        """Class constructor"""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """tokenize"""
        all_sentences = [(pt.numpy().decode("utf-8"), en.numpy().decode(
            "utf-8"))
                         for pt, en in data]
        pt_texts, en_texts = (list(pair) for pair in zip(
            *all_sentences)) if all_sentences else ([], [])

        base_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True,
            clean_up_tokenization_spaces=True
        )
        base_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True,
            clean_up_tokenization_spaces=True
        )

        new_pt = base_pt.train_new_from_iterator(pt_texts, vocab_size=2**13)
        new_en = base_en.train_new_from_iterator(en_texts, vocab_size=2**13)
        return new_pt, new_en

    def encode(self, pt, en):
        """encode"""
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')
        pt_start = len(self.tokenizer_pt)
        pt_end = len(self.tokenizer_pt) + 1
        en_start = len(self.tokenizer_en)
        en_end = len(self.tokenizer_en) + 1

        pt_token_ids = self.tokenizer_pt.encode(pt_text,
                                             add_special_tokens=False)
        en_token_ids = self.tokenizer_en.encode(en_text,
                                             add_special_tokens=False)

        encoded_pt = [pt_start] + pt_token_ids + [pt_end]
        encoded_en = [en_start] + en_token_ids + [en_end]

        return encoded_pt, encoded_en