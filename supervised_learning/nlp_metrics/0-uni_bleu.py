#!/usr/bin/env python3
"""0. Unigram BLEU score"""
from collections import Counter
import numpy as np


def uni_bleu(references, sentence):
    """ calculates the unigram BLEU score for a sentence:
    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    references = [["the", "cat", "is", "on", "the", "mat"],
    ["there", "is", "a", "cat", "on", "the", "mat"]]
    sentence = ["there", "is", "a", "cat", "here"]
    Returns: the unigram BLEU score"""
    # occurrences of each word in sentence
    sentence_counts = Counter(sentence)
    # Counter({'there': 1, 'is': 1, 'a': 1, 'cat': 1, 'here': 1})
    max_ref_counts = {}
    for reference in references:
        reference_counts = Counter(reference)
        for word in sentence_counts:
            max_ref_counts[word] = max(max_ref_counts.get(word, 0),
                                       reference_counts.get(word, 0))
            # {'there': 1, 'is': 1, 'a': 1, 'cat': 1, 'here': 0}
    # Calculate the clipped count for the proposed sentence
    count_clip = sum(min(sentence_counts[word],
                         max_ref_counts.get(word, 0)) for word in sentence_counts)
    # 4
    # precision
    precision = count_clip / len(sentence)
    # 0.8
    # Find the closest reference length
    sentence_length = len(sentence)
    closest_ref_len = min((abs(len(ref) - sentence_length), len(ref))
                          for ref in references)[1]

    if sentence_length > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_length)
    bleu_score = brevity_penalty * precision
    return bleu_score
