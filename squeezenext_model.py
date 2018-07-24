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
        """
        Define batch size of dictionary
        :param features:
            Feature dict
        :param labels:
            Labels dict
        :return:
            (features,label)
        """
        features = tools.define_first_dim(features, self.batch_size)
        labels = tools.define_first_dim(labels, self.batch_size)
        return (features, labels)

    def input_fn(self, file_pattern,training):
        """
        Input fn of model
        :param file_pattern:
            Glob file pattern
        :param training:
            Whether or not the model is training
        :return:
            Input generator
        """
        return self.define_batch_size(*self.read_tf_records(file_pattern,training=training).get_next())

    def model_fn(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """

        training = mode == tf.estimator.ModeKeys.TRAIN
        # init model class
        model = squeezenext.SqueezeNext(self.num_classes, params["block_defs"], params["input_def"], params["groups"],params["seperate_relus"])
        # create model inside the argscope of the model
        with slim.arg_scope(squeezenext.squeeze_next_arg_scope(training)):
            predictions,endpoints = model(features["image"], training)

        # output predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': tf.argmax(tf.nn.softmax(predictions), -1),
                'probabilities': tf.nn.softmax(predictions),
                'logits': predictions,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # create loss (should be equal to caffe softmaxwithloss)
        loss = tf.losses.softmax_cross_entropy(tf.squeeze(labels["class_vec"],axis=1), predictions)

        # create histogram of class spread
        tf.summary.histogram("classes",labels["class_idx"])

        if training:
            # init poly optimizer
            optimizer = PolyOptimizer(params)
            # define train op
            train_op = optimizer.optimize(loss, training, params["total_steps"])

            # if params["output_train_images"] is true output images during training
            if params["output_train_images"]:
                tf.summary.image("training", features["image"])

            # create estimator training spec, which also outputs the model_stats of the model to params["model_dir"]
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[
                tools.stats._ModelStats("squeezenext", params["model_dir"],
                                        features["image"].get_shape().as_list()[0])])



        if mode == tf.estimator.ModeKeys.EVAL:
            # Define the metrics:
            metrics_dict = {
                'Recall@1': tf.metrics.accuracy(tf.argmax(predictions, axis=-1), labels["class_idx"][:, 0]),
                'Recall@5': metrics.streaming_sparse_recall_at_k(predictions, tf.cast(labels["class_idx"], tf.int64),
                                                                 5)
            }
            # output eval images
            eval_summary_hook = tf.train.SummarySaverHook(
                save_steps=100,
                output_dir=os.path.join(params["model_dir"],"eval"),
                summary_op=tf.summary.image("validation", features["image"]))

            #return eval spec
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics_dict,
                evaluation_hooks=[eval_summary_hook])
