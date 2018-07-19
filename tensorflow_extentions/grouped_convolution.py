import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
import collections

slim = tf.contrib.slim

def grouped_convolution2D(inputs,filters,padding,num_groups,
        strides=None,
        dilation_rate=None,
        name="grouped_convolutions"):
    input_list = tf.split(inputs, num_groups,axis=-1)
    filter_list = tf.split(filters, num_groups,axis=-1)
    output_list = []
    for conv_idx,(input_tensor,filter_tensor) in enumerate(zip(input_list,filter_list)):
        output_list.append(tf.nn.convolution(
            input_tensor,
            filter_tensor,
            padding,
            strides=strides,
            dilation_rate=dilation_rate,
            name="grouped_convolution_{}".format(conv_idx)
        ))
    outputs = tf.concat(output_list,axis=-1)
    return outputs

@slim.add_arg_scope
def grouped_convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                rate=1,
                groups=1,
                activation_fn=tf.nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=tf.initializers.zeros(),
                biases_regularizer=None,
                reuse=None,
                trainable=True,
                scope=None,
                outputs_collections=None):

    # if no group size specified or less than/equal to zero return a normal convolution
    if groups == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size,stride=stride,padding=padding,activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=weights_initializer,
                weights_regularizer=weights_regularizer,
                biases_initializer=biases_initializer,
                biases_regularizer=biases_regularizer,
                reuse=reuse,
                trainable=trainable,
                scope=scope)

    assert groups > 1, "Specify a number of groups greater than zero, groups given is {}".format(groups)
    input_channels = inputs.get_shape().as_list()[-1]
    lowest_channels = min(input_channels, num_outputs)
    assert lowest_channels%groups == 0, "the remainder of min(input_channels,output_channels)/group_size should be zero"
    group_size = lowest_channels/groups
    assert max(input_channels, num_outputs)%group_size == 0, "the remainder of max(input_channels,output_channels)/group_size should be zero"

    with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse) as sc:
        if isinstance(kernel_size, collections.Iterable):
            weights_shape = kernel_size+[group_size]+[num_outputs]
        else:
            weights_shape = [kernel_size,kernel_size,group_size,num_outputs]
        weights = slim.variable('weights',
                                shape=weights_shape,
                                initializer=weights_initializer,
                                regularizer=weights_regularizer,
                                device='/CPU:0')
        strides = [stride,stride]
        dilation_rate = [rate,rate]
        outputs = grouped_convolution2D(inputs,weights,padding,groups,
                    strides=strides,
                    dilation_rate=dilation_rate)
        if biases_initializer is not None:
            biases = slim.variable('biases',
                                    shape=[num_outputs],
                                    initializer=biases_initializer,
                                    regularizer=biases_regularizer,
                                    device='/CPU:0')
            outputs = tf.nn.bias_add(outputs,biases)
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)