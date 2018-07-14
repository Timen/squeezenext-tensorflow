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
