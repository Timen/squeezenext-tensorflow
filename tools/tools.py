from __future__ import absolute_import

import tensorflow as tf
import os

def define_first_dim(tensor_dict,dim_size):
    for key, tensor in tensor_dict.iteritems():
        shape = tensor.get_shape().as_list()[1:]
        tensor_dict[key] = tf.reshape(tensor, [dim_size] + shape)
    return tensor_dict

def get_checkpoint_step(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt is None:
        return None
    else:
        return int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])


def Print(tensor):
    def my_func(x):
        print(x.shape)
        print(x)
        return x

    return tf.py_func(my_func, [tensor], tensor.dtype)

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
