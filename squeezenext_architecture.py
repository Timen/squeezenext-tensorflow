from __future__ import absolute_import

import tensorflow as tf

slim = tf.contrib.slim

import tensorflow_extentions as tfe


def squeezenext_unit(inputs, filters, stride,height_first_order, groups,seperate_relus):
    """
    Squeezenext unit according to:
    https://arxiv.org/pdf/1803.10615.pdf

    :param inputs:
        Input tensor
    :param filters:
        Number of filters at output of this unit
    :param stride:
        Input stride
    :param height_first_order:
        Whether to first perform seperable convolution in the vertical direcation or horizontal direction
    :param groups:
        Number of groups for some of the convolutions (which ones are different from the paper but equal to:
        https://github.com/amirgholami/SqueezeNext/blob/master/1.0-G-SqNxt-23/train_val.prototxt)
    :return:
        Output tensor, not(height_first_order)
    """
    input_channels = inputs.get_shape().as_list()[-1]
    shortcut = inputs
    out_activation = tf.nn.relu if bool(seperate_relus) else None
    # shorcut convolution only to be executed if input channels is different from output channels or
    # stride is greater than 1.
    if input_channels != filters or stride != 1:
        shortcut = slim.conv2d(shortcut, filters, [1, 1], stride=stride, activation_fn=out_activation)

    # input 1x1 reduction convolutions
    block = tfe.grouped_convolution(inputs, filters / 2, [1, 1], groups, stride=stride)
    block = slim.conv2d(block, block.get_shape().as_list()[-1] / 2, [1, 1])

    # seperable convolutions
    if height_first_order:
        input_channels_seperated = block.get_shape().as_list()[-1]
        block = tfe.grouped_convolution(block, input_channels_seperated * 2, [3, 1], groups)
        block = tfe.grouped_convolution(block, block.get_shape().as_list()[-1], [1, 3], groups)

    else:
        input_channels_seperated = block.get_shape().as_list()[-1]
        block = tfe.grouped_convolution(block, input_channels_seperated * 2, [1, 3], groups)
        block = tfe.grouped_convolution(block, block.get_shape().as_list()[-1], [3, 1], groups)
    # switch order next unit
    height_first_order = not height_first_order

    # output convolutions
    block = slim.conv2d(block, block.get_shape().as_list()[-1] * 2, [1, 1],activation_fn=out_activation)
    assert block.get_shape().as_list()[-1] == filters, "Block output channels not equal to number of specified filters"


    return tf.nn.relu(block + shortcut),height_first_order


class SqueezeNext(object):
    """Base class for building the SqueezeNext Model."""

    def __init__(self, num_classes, block_defs, input_def,groups,seperate_relus):
        self.num_classes = num_classes
        self.block_defs = block_defs
        self.input_def = input_def
        self.groups = groups
        self.seperate_relus = seperate_relus


    def __call__(self, inputs, training,height_first_order = True):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        with tf.variable_scope("squeezenext"):
            input_filters, input_kernel,input_stride = self.input_def
            endpoints = {}

            # input convolution and pooling
            net = slim.conv2d(inputs, input_filters, input_kernel, stride=input_stride,scope="input_conv",padding="VALID")
            endpoints["input_conv"] = net
            net = slim.max_pool2d(net, [3, 3], stride=2)
            endpoints["max_pool"] = net

            # create block based network
            for block_idx,block_def in enumerate(self.block_defs):

                filters,units,stride = block_def
                with tf.variable_scope("block_{}".format(block_idx)):
                    # create seperate units inside a block
                    for unit_idx in range(0,units):
                        with tf.variable_scope("unit_{}".format(unit_idx)):
                            if unit_idx != 0:
                                # perform striding only in first unit of a block
                                net,height_first_order = squeezenext_unit(net,filters,1,height_first_order,self.groups,self.seperate_relus)
                            else:
                                net,height_first_order = squeezenext_unit(net, filters, stride,height_first_order,self.groups,self.seperate_relus)
                        endpoints["block_{}".format(block_idx)+"/"+"unit_{}".format(unit_idx)]=net
            # output conv and pooling
            net = slim.conv2d(net, 128, [1,1],scope="output_conv")
            endpoints["output_conv"] = net
            net = tf.squeeze(slim.avg_pool2d(net,net.get_shape().as_list()[1:3],scope="avg_pool_out", padding="VALID"),axis=[1,2])
            endpoints["avg_pool_out"] = net

            # Fully connected output without biases
            output = slim.fully_connected(net,self.num_classes,activation_fn=None,normalizer_fn=None, biases_initializer=None)
            endpoints["output"] = output

        return output,endpoints



def squeeze_next_arg_scope(is_training,
                           weight_decay=0.0001):
    """
    Setup slim arg scope according to paper and github project
    :param is_training:
        Whether or not the network is training
    :param weight_decay:
        Weight decay of the convolutional layers
    :return:
        Slim arg scope
    """
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.999,
        'epsilon': 1e-5,
        'fused': True,
    }

    # Use xavier an l2 decay
    weights_init = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)


    with slim.arg_scope([slim.conv2d,tfe.grouped_convolution],
                        weights_initializer=weights_init,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        # No biases in the convolutions (are already included in batch_norm)
                        biases_initializer=None,
                        weights_regularizer=regularizer):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as sc:
            return sc
