#!/usr/bin/env python3
"""1. n_gram BLEU score"""
from collections import Counter
import numpy as np


def generate_ngram(sentence, order):
    """generate n-grams from a sentence"""
    sentence_ngrams = [' '.join(
        sentence[i:i + order])for i in range(len(sentence) - order + 1)]
    return sentence_ngrams


def ngram_bleu(references, sentence, n):
    """calculates the n-gram BLEU score for a sentence:
    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    n is the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score"""
    sentence_ngrams = generate_ngram(sentence, n)
    s_counts = Counter(sentence_ngrams)
    max_ref_counts = {}
    for reference in references:
        reference_ngrams = generate_ngram(reference, n)
        reference_counts = Counter(reference_ngrams)
        for ngram in s_counts:
            max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0),
                                        reference_counts.get(ngram, 0))
    count_clip = sum(min(s_counts[ngram],
                         max_ref_counts.get(ngram, 0)) for ngram in s_counts)
    precision = count_clip / len(sentence_ngrams)
    sentence_length = len(sentence)
    closest_ref_len = min((abs(len(ref) - sentence_length), len(ref))
                          for ref in references)[1]

    # Calculate brevity penalty
    if sentence_length > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_length)
    bleu_score = brevity_penalty * precision
    return bleu_score
