from __future__ import absolute_import

import tensorflow as tf

slim = tf.contrib.slim
metrics = tf.contrib.metrics
import squeezenext_architecture as squeezenext
from optimizer  import PolyOptimizer
from dataloader import ReadTFRecords
import tools
import os
metrics = tf.contrib.metrics

class Model(object):
    def __init__(self, config, batch_size):
        self.image_size = config["image_size"]
        self.num_classes = config["num_classes"]
        self.batch_size = batch_size
        self.read_tf_records = ReadTFRecords(self.image_size, self.batch_size, self.num_classes)

    def define_batch_size(self, features, labels):
        features = tools.define_first_dim(features, self.batch_size)
        labels = tools.define_first_dim(labels, self.batch_size)
        return (features, labels)

    def input_fn(self, file_pattern,training):
        return self.define_batch_size(*self.read_tf_records(file_pattern,training=training).get_next())

    def model_fn(self, features, labels, mode, params):
        training = mode == tf.estimator.ModeKeys.TRAIN
        model = squeezenext.SqueezeNext(self.num_classes, params["block_defs"], params["input_def"], params["groups"])

        with slim.arg_scope(squeezenext.squeeze_next_arg_scope(training)):
            predictions,endpoints = model(features["image"], training)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': tf.argmax(tf.nn.softmax(predictions), -1),
                'probabilities': tf.nn.softmax(predictions),
                'logits': predictions,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.softmax_cross_entropy(tf.squeeze(labels["class_vec"],axis=1), predictions)

        tf.summary.histogram("classes",labels["class_idx"])

        if training:
            optimizer = PolyOptimizer(params)
            train_op = optimizer.optimize(loss, training, params["total_steps"])

            if params["output_train_images"]:
                tf.summary.image("training", features["image"])
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[
                tools.stats._ModelStats("squeezenext", params["model_dir"],
                                        features["image"].get_shape().as_list()[0])])



        if mode == tf.estimator.ModeKeys.EVAL:
            # Define the metrics:
            metrics_dict = {
                'Accuracy': tf.metrics.accuracy(tf.argmax(predictions, axis=-1), labels["class_idx"][:, 0]),
                'Recall@5': metrics.streaming_sparse_recall_at_k(predictions, tf.cast(labels["class_idx"], tf.int64),
                                                                 5)
            }
            eval_summary_hook = tf.train.SummarySaverHook(
                save_steps=100,
                output_dir=os.path.join(params["model_dir"],"eval"),
                summary_op=tf.summary.image("validation", features["image"]))
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics_dict,
                evaluation_hooks=[eval_summary_hook])
