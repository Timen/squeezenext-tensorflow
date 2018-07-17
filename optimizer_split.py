from __future__ import absolute_import

import tensorflow as tf
slim = tf.contrib.slim


def get_or_create_global_step():
    global_step = tf.train.get_global_step()
    if global_step is None:
        global_step = tf.train.create_global_step()
    return global_step

def warmup_phase(learning_rate_schedule,base_lr,warmup_steps,warmup_learning_rate):
    with tf.name_scope("warmup_learning_rate"):
        global_step = tf.cast(get_or_create_global_step(),tf.float32)
        if warmup_steps > 0:
            if base_lr < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                                 'warmup_learning_rate.')
            slope = (base_lr - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            learning_rate_schedule = tf.where(global_step < warmup_steps, warmup_rate,
                                     learning_rate_schedule)
        return learning_rate_schedule

class PolyOptimizer(object):
    def __init__(self, training_params):
        self.base_lr = training_params["base_lr"]
        self.warmup_steps = 780
        self.warmup_learning_rate = 0.1
        self.power = 2.0
        self.momentum = 0.9
        self.gradient_multiplier = 32

    def optimize(self,loss, training,total_steps):
        with tf.name_scope("PolyOptimizer"):
            global_step = get_or_create_global_step()

            learning_rate_schedule = tf.train.polynomial_decay(
                learning_rate=self.base_lr,
                global_step=global_step,
                decay_steps=total_steps,
                power=self.power
            )
            learning_rate_schedule = warmup_phase(learning_rate_schedule,self.base_lr, self.warmup_steps,self.warmup_learning_rate)
            tf.summary.scalar("learning_rate",learning_rate_schedule)
            optimizer = tf.train.MomentumOptimizer(learning_rate_schedule,self.momentum)
            gradient_multipliers = {}
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name[:-2],var)
                # paper describes the gradient being the sum of the 32 different workers
                # this attempts to replicate that by multiplying the gradients by 32.
                gradient_multipliers[var.name[:-2]] = self.gradient_multiplier
            return slim.learning.create_train_op(loss,
                                        optimizer,
                                global_step=global_step,
                                summarize_gradients=True,
                                gradient_multipliers=gradient_multipliers,
                                update_ops=None if training else []),optimizer




