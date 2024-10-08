#!/usr/bin/env python3
"""6. Train"""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """ builds, trains, and saves a neural network classifier"""
    m, nx = X_train.shape
    classes = Y_train.shape[1]
    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # add tensors and operations to collections for later retrieval
        tf.add_to_collection('x', x)
        tf.add_to_collection('y', y)
        tf.add_to_collection('y_pred', y_pred)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('accuracy', accuracy)
        tf.add_to_collection('train_op', train_op)

        for i in range(iterations + 1):
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            train_loss, train_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})

            valid_loss, valid_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            if i == iterations or i % 100 == 0:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_loss}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_loss}")
                print(f"\tValidation Accuracy: {valid_acc}")

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        save_path = saver.save(sess, save_path)

    return save_path
