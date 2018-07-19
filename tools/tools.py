from __future__ import absolute_import

import tensorflow as tf
import os

def define_first_dim(tensor_dict,dim_size):
    """
    Define the first dim keeping the remaining dims the same
    :param tensor_dict:
        Dictionary of tensors
    :param dim_size:
        Size of first dimention
    :return:
        Dictionary of dimensions with the first dim defined as dim_size
    """
    for key, tensor in tensor_dict.iteritems():
        shape = tensor.get_shape().as_list()[1:]
        tensor_dict[key] = tf.reshape(tensor, [dim_size] + shape)
    return tensor_dict

def get_checkpoint_step(checkpoint_dir):
    """
    Get step at which checkpoint was saved from file name
    :param checkpoint_dir:
        Directory containing a checkpoint
    :return:
        Step at which checkpoint was saved
    """
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt is None:
        return None
    else:
        return int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])


def get_or_create_global_step():
    """
    Checks if global step variable exists otherwise creates it
    :return:
    Global step tensor
    """
    global_step = tf.train.get_global_step()
    if global_step is None:
        global_step = tf.train.create_global_step()
    return global_step

def warmup_phase(learning_rate_schedule,base_lr,warmup_steps,warmup_learning_rate):
    """
    Ramps up the learning rate from warmup_learning_rate till base_lr in warmup_steps before
    switching to learning_rate_schedule.
    The warmup is linear and calculated using the below functions.
    slope = (base_lr - warmup_learning_rate) / warmup_steps
    warmup_rate = slope * global_step + warmup_learning_rate

    :param learning_rate_schedule:
        A regular learning rate schedule such as stepwise,exponential decay etc
    :param base_lr:
        The learning rate to which to ramp up to
    :param warmup_steps:
        The number of steps of the warmup phase
    :param warmup_learning_rate:
        The learning rate from which to start ramping up to base_lr
    :return:
        Warmup learning rate for global step <  warmup_steps else returns learning_rate_schedule
    """
    with tf.name_scope("warmup_learning_rate"):
        global_step = tf.cast(get_or_create_global_step(),tf.float32)
        if warmup_steps > 0:
            if base_lr < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                                 'warmup_learning_rate.')
            slope = (learning_rate_schedule - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            learning_rate_schedule = tf.where(global_step < warmup_steps, warmup_rate,
                                     learning_rate_schedule)
        return learning_rate_schedule
