#!/usr/bin/env python3
"""mmkjkj"""
import numpy as np

def create_confusion_matrix(labels, logits):
    """the function compares the predicted labels derived from
    the logits with the true labels, and updates the confusion
    matrix accordingly by incrementing the appropriate cell
    for each pair of predicted and true labels."""
    m, classes = labels.shape
    confusion = np.zeros((classes, classes), dtype=np.int32)
    
    predicted_labels = np.argmax(logits, axis=1)
    correct_labels = np.argmax(labels, axis=1)
    
    for i in range(m):
        confusion[correct_labels[i], predicted_labels[i]] += 1
        
    return confusion.astype(np.float64)
