from __future__ import absolute_import

import tensorflow as tf
slim = tf.contrib.slim

def top_k_accuracy(predictions,labels,k):
    """
    Function to calculate the top k accuracy of the predictions
    :param predictions:
        A tensor of shape [batch,1,num_classes] with the raw output of the last layer of the network
    :param labels:
        A tensor of shape [batch] with the index value of the class (1..num_classes)
    :param k:
        The scalar to determine how many entries of the probability sorted list to consider for a matched prediction
    :return:
        A mean metrics op of the accuracy and the accuracy of that batch
    """
    with tf.variable_scope("top_{}_accuracy".format(k)):
        # Apply softmax to predictions, possibly unnecessary but keeping for clarity
        softmax_predictions = tf.nn.softmax(predictions)
        # Calculate average top k accuracy
        accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(softmax_predictions,labels[:,0],k), tf.float32))
        return tf.metrics.mean(accuracy),accuracy

