#!/usr/bin/env python3
"""1. n_gram BLEU score"""
from collections import Counter
import numpy as np


def generate_ngram(sentence, order):
    """generate n-grams from a sentence"""
    sentence_ngrams = [' '.join(
        sentence[i:i + order])for i in range(len(sentence) - order + 1)]
    return sentence_ngrams


def cumulative_bleu(references, sentence, n):
    """calculates the cumulative n-gram BLEU score for a sentence:
    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    n is the size of the largest n-gram to use for evaluation
    All n-gram scores should be weighted evenly
    Returns: the cumulative n-gram BLEU score"""
    precisions = []
    for i in range(1, n + 1):
        s_ngrams = generate_ngram(sentence, i)
        s_counts = Counter(s_ngrams)
        max_ref_counts = {}
        for ref in references:
            ref_ngrams = generate_ngram(ref, i)
            ref_counts = Counter(ref_ngrams)
            for ngram in s_counts:
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0),
                                            ref_counts.get(ngram, 0))
        count_clip = sum(min(s_counts[ngram],
                             max_ref_counts.get(ngram,
                                                0)) for ngram in s_counts)
        if len(s_ngrams) > 0:
            precisions.append(count_clip / len(s_ngrams))
        else:
            precisions.append(0)
    if all(precisions):
        precision_product = np.prod(precisions)
        precision_geom_mean = precision_product ** (1 / n)
    else:
        precision_geom_mean = 0
    sentence_length = len(sentence)
    closest_ref_len = min((abs(len(ref) - sentence_length), len(ref))
                          for ref in references)[1]
    # Calculate brevity penalty
    if sentence_length > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_length)
    bleu_score = brevity_penalty * precision_geom_mean
    return bleu_score
