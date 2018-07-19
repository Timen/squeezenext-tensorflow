from __future__ import absolute_import

import tensorflow as tf
import tools
slim = tf.contrib.slim



class PolyOptimizer(object):
    def __init__(self, training_params):
        self.base_lr = training_params["base_lr"]
        self.warmup_steps = training_params["warmup_iter"]
        self.warmup_learning_rate = training_params["warmup_start_lr"]
        self.power = 2.0
        self.momentum = 0.9

    def optimize(self,loss, training,total_steps):
        with tf.name_scope("PolyOptimizer"):
            global_step = tools.get_or_create_global_step()

            learning_rate_schedule = tf.train.polynomial_decay(
                learning_rate=self.base_lr,
                global_step=global_step,
                decay_steps=total_steps,
                power=self.power
            )
            learning_rate_schedule = tools.warmup_phase(learning_rate_schedule,self.base_lr, self.warmup_steps,self.warmup_learning_rate)
            tf.summary.scalar("learning_rate",learning_rate_schedule)
            optimizer = tf.train.MomentumOptimizer(learning_rate_schedule,self.momentum)
            return slim.learning.create_train_op(loss,
                                        optimizer,
                                global_step=global_step,
                                aggregation_method=tf.AggregationMethod.ADD_N,
                                update_ops=None if training else [])




