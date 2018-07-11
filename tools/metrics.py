import tensorflow as tf
slim = tf.contrib.slim

def top_k_accuracy(predictions,labels,k):
    softmax_predictions = tf.nn.softmax(predictions)
    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(softmax_predictions,labels[:,0],k), tf.float32))
    return tf.metrics.mean(accuracy),accuracy

