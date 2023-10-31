#!/usr/bin/env python3
"""7. Evaluate"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """evaluates the output of a neural network"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        feed_dict = {x: X, y: Y}
        Y_pred, acc, cost = sess.run([pred, accuracy, loss], feed_dict=feed_dict)

    return Y_pred, acc, cost
