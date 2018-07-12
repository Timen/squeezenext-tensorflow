from __future__ import absolute_import

import tensorflow as tf

def define_first_dim(tensor_dict,dim_size):
    for key, tensor in tensor_dict.iteritems():
        shape = tensor.get_shape().as_list()[1:]
        tensor_dict[key] = tf.reshape(tensor, [dim_size] + shape)
    return tensor_dict