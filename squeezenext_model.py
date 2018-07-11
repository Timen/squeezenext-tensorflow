import tensorflow as tf

slim = tf.contrib.slim
import squeezenext_architecture as squeezenext
from optimizer import PolyOptimizer
from dataloader import ReadTFRecords
import tools


class Model(object):
    def __init__(self, config, batch_size):
        self.image_size = config["image_size"]
        self.num_classes = config["num_classes"]
        self.batch_size = batch_size
        self.read_tf_records = ReadTFRecords(self.image_size, self.batch_size, self.num_classes, config["mean_value"])

    def define_batch_size(self, features, labels):
        features = tools.define_first_dim(features, self.batch_size)
        labels = tools.define_first_dim(labels, self.batch_size)
        return (features, labels)

    def input_fn(self, file_pattern):
        return self.define_batch_size(*self.read_tf_records(file_pattern).get_next())

    def model_fn(self, features, labels, mode, params):
        training = mode == tf.estimator.ModeKeys.TRAIN
        model = squeezenext.SqueezeNext(self.num_classes, params["block_defs"], params["input_def"])

        with slim.arg_scope(squeezenext.squeeze_next_arg_scope(training)):
            predictions = model(features["image"], training)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': tf.argmax(predictions, -1),
                'probabilities': tf.nn.softmax(predictions),
                'logits': predictions,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.softmax_cross_entropy(labels["class_vec"], tf.expand_dims(predictions, axis=1))

        top_1_accuracy_metric,top_1_accuracy = tools.metrics.top_k_accuracy(predictions, labels["class_idx"], 1)
        top_5_accuracy_metric,top_5_accuracy = tools.metrics.top_k_accuracy(predictions, labels["class_idx"], 5)
        tf.summary.scalar("top_1_accuracy", top_1_accuracy)
        tf.summary.scalar("top_5_accuracy", top_5_accuracy)

        if training:
            optimizer = PolyOptimizer(params)
            train_op = optimizer.optimize(loss, training)
            if params["output_train_images"]:
                tf.summary.image("training", features["image"])
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[
                tools.stats._ModelStats("squeezenext", params["model_dir"],
                                        features["image"].get_shape().as_list()[0])])

        metrics = {"top_1_accuracy": top_1_accuracy_metric, "top_5_accuracy": top_5_accuracy_metric}
        if mode == tf.estimator.ModeKeys.EVAL:
            images_summary = tf.summary.image("validation", features["image"])
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=tf.train.SummarySaverHook(summary_op=images_summary))
