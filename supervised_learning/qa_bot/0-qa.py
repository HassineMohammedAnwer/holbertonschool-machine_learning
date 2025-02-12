#!/usr/bin/env python3
"""0. Question Answering"""

import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow as tf


def question_answer(question, reference):
    """ qa """
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word'
                                              '-masking-finetuned-squad')
    model = (
        hub.load("https://www.kaggle.com/models/seesee/bert/"
                 "TensorFlow2/uncased-tf2-qa/1"))

    question_t = tokenizer.tokenize(question)
    reference_t = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + question_t + ['[SEP]'] + reference_t + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)
    token_type_ids = [0] * (len(question_t) + 2) + [1] * (len(reference_t) + 1)

    def to_tensor(values):
        """dsfvsfd"""
        return tf.expand_dims(tf.convert_to_tensor(
            values, dtype=tf.int32), axis=0)

    input_word_ids = to_tensor(input_ids)
    input_mask_tensor = to_tensor(input_mask)
    input_type_ids_tensor = to_tensor(token_type_ids)

    outputs = model([input_word_ids, input_mask_tensor, input_type_ids_tensor])
    start_logits, end_logits = outputs[0], outputs[1]

    start_index = tf.argmax(start_logits[0][1:]) + 1
    end_index = tf.argmax(end_logits[0][1:]) + 1

    answer_tokens = tokens[start_index: end_index + 1]
    if not answer_tokens:
        return None

    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer
