#!/usr/bin/env python3
"""0. Bag Of Words"""
import numpy as np
import re


def clean_and_tokenize(sentence):
    """Cleans and tokenizes"""
    sent = re.sub(r"\'s\b", "", sentence)
    cleaned = re.sub(r'[^a-zA-Z\s]', '', sent).lower()
    tokens = cleaned.split()    
    return tokens

def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix:
    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
    If None, all words within sentences should be used
    Returns: embeddings, features
    embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
    s is the number of sentences in sentences
    f is the number of features analyzed
    features is a list of the features used for embeddings
    You are not allowed to use genism library."""
    sentences_l = [clean_and_tokenize(sentence) for sentence in sentences]
    # print('sentences_l')
    # print(sentences_l)
    if vocab is None:
        vocab = np.array(sorted({word for sentence in sentences_l for word in sentence}))
    num_sentences = len(sentences_l)
    num_features = len(vocab)
    embedding_matrix = np.zeros((num_sentences, num_features), dtype=int)
    word_md = {word: i for i, word in enumerate(vocab)}
    # print(word_md)
    for i, sentence in enumerate(sentences_l):
        for word in sentence:
            if word in word_md:
                embedding_matrix[i, word_md[word]] += 1
    return embedding_matrix, vocab
