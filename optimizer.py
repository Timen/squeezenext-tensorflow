import tensorflow as tf
slim = tf.contrib.slim



class PolyOptimizer(object):
    def __init__(self, training_params):
        self.learning_rate = training_params["base_lr"]
        self.decay_steps = 150136
        self.warmup_steps = 780
        self.warmup_learning_rate = 0.1
        self.power = 2.0
        self.end_learning_rate = 0.0000001
        self.momentum = 0.9

    def optimize(self,loss, training):
        global_step = tf.train.get_global_step()
        if global_step is None:
            global_step = tf.train.create_global_step()
        learning_rate = tf.train.polynomial_decay(
            learning_rate=self.learning_rate,
            global_step=global_step,
            decay_steps=self.decay_steps,
            end_learning_rate=self.end_learning_rate,
            power=self.power
        )
        if self.warmup_steps > 0:
            if  self.learning_rate < self.warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                                 'warmup_learning_rate.')
            slope = (self.learning_rate - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * tf.cast(global_step,
                                          tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(global_step < self.warmup_steps, warmup_rate,
                                     learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate,self.momentum)
        return slim.learning.create_train_op(loss,
                                    optimizer,
                            global_step=global_step,
                            update_ops=None if training else [])




