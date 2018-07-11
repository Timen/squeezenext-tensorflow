import tensorflow as tf
slim = tf.contrib.slim
import squeezenext_architecture as squeezenext
from optimizer import PolyOptimizer
import metrics
import tools.model_stats as stats

class Model(object):
    def __init__(self,config):
        self.image_size = config["image_size"]
        self.num_classes = config["num_classes"]

    def input_fn(self, file_pattern,batch_size):
        features_dummy = {"image":tf.ones([batch_size,self.image_size ,self.image_size ,3])}
        labels_dummy = {"class_idx": tf.ones([batch_size, 1]),"class_vec": tf.one_hot(tf.ones([batch_size, 1],dtype=tf.int32),self.num_classes)}
        return (features_dummy,labels_dummy)

    def model_fn(self,features,labels,mode,params):
        training = mode == tf.estimator.ModeKeys.TRAIN
        model = squeezenext.SqueezeNext(self.num_classes,params["block_defs"],params["input_def"])
        with slim.arg_scope(squeezenext.squeeze_next_arg_scope(training)):
            predictions = model(features["image"],training)

        loss =  tf.losses.softmax_cross_entropy(labels["class_vec"],tf.expand_dims(predictions,axis=1))



        if training:
            optimizer = PolyOptimizer(params)
            train_op =  optimizer.optimize(loss,training)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,training_hooks=[stats._ModelStats("squeezenext",params["model_dir"],features["image"].get_shape().as_list()[0])])

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': tf.argmax(predictions, -1),
                'probabilities': tf.nn.softmax(predictions),
                'logits': predictions,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        top_1_accuracy_metric = metrics.top_k_accuracy(predictions,labels["class"],1)
        top_5_accuracy_metric = metrics.top_k_accuracy(predictions, labels["class"], 5)

        metrics = {"top_1_accuracy":top_1_accuracy_metric,"top_5_accuracy":top_5_accuracy_metric}
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)