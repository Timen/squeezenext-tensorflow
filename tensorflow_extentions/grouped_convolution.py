import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
import collections

slim = tf.contrib.slim


def grouped_convolution2D(inputs, filters, padding, num_groups,
                          strides=None,
                          dilation_rate=None):
    """
    Performs a grouped convolution by applying a normal convolution to each of the seperate groups
    :param inputs:
        Input of the shape [<batch_size>,H,W,inC]
    :param filters:
        [H,W,inC/num_groups,outC]
    :param padding:
        What padding to use
    :param num_groups:
        Number of seperate groups
    :param strides:
        Stride
    :param dilation_rate:
        Dilation rate
    :return:
        Output of shape [<batch_size>,H/stride,W/stride,outC]
    """
    # Split input and outputs along their last dimension
    input_list = tf.split(inputs, num_groups, axis=-1)
    filter_list = tf.split(filters, num_groups, axis=-1)
    output_list = []

    # Perform a normal convolution on each split of the input and filters
    for conv_idx, (input_tensor, filter_tensor) in enumerate(zip(input_list, filter_list)):
        output_list.append(tf.nn.convolution(
            input_tensor,
            filter_tensor,
            padding,
            strides=strides,
            dilation_rate=dilation_rate,
            name="grouped_convolution" + "_{}".format(conv_idx)
        ))
    # Concatenate ouptputs along there last dimentsion
    outputs = tf.concat(output_list, axis=-1)

    return outputs


@slim.add_arg_scope
def grouped_convolution(inputs,
                        num_outputs,
                        kernel_size,
                        groups,
                        stride=1,
                        padding='SAME',
                        rate=1,
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
    """Adds an 2-D grouped convolution followed by an optional batch_norm layer.
      `convolution` creates a variable called `weights`, representing the
      convolutional kernel, that is convolved (actually cross-correlated) with the
      `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
      provided (such as `batch_norm`), it is then applied. Otherwise, if
      `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
      variable would be created and added the activations. Finally, if
      `activation_fn` is not `None`, it is applied to the activations as well.
      Performs atrous convolution with input stride/dilation rate equal to `rate`
      if a value > 1 for any dimension of `rate` is specified.  In this case
      `stride` values != 1 are not supported.
      Args:
        inputs: A Tensor of rank N+2 of shape
          `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
          not start with "NC" (default), or
          `[batch_size, in_channels] + input_spatial_shape` if data_format starts
          with "NC".
        num_outputs: Integer, the number of output filters.
        kernel_size: A sequence of N positive integers specifying the spatial
          dimensions of the filters.  Can be a single integer to specify the same
          value for all spatial dimensions.
        groups: Number of groups to split the input up in before applying convolutions to the
            seperate groups. If groups==1 return normal slim.conv2d.
        stride: A sequence of N positive integers specifying the stride at which to
          compute output.  Can be a single integer to specify the same value for all
          spatial dimensions.  Specifying any `stride` value != 1 is incompatible
          with specifying any `rate` value != 1.
        padding: One of `"VALID"` or `"SAME"`.
        rate: A sequence of N positive integers specifying the dilation rate to use
          for atrous convolution.  Can be a single integer to specify the same
          value for all spatial dimensions.  Specifying any `rate` value != 1 is
          incompatible with specifying any `stride` value != 1.
        activation_fn: Activation function. The default value is a ReLU function.
          Explicitly set it to None to skip it and maintain a linear activation.
        normalizer_fn: Normalization function to use instead of `biases`. If
          `normalizer_fn` is provided then `biases_initializer` and
          `biases_regularizer` are ignored and `biases` are not created nor added.
          default set to None for no normalizer function
        normalizer_params: Normalization function parameters.
        weights_initializer: An initializer for the weights.
        weights_regularizer: Optional regularizer for the weights.
        biases_initializer: An initializer for the biases. If None skip biases.
        biases_regularizer: Optional regularizer for the biases.
        reuse: Whether or not the layer and its variables should be reused. To be
          able to reuse the layer scope must be given.
        outputs_collections: Collection to add the outputs.
        trainable: If `True` also add variables to the graph collection
          `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        scope: Optional scope for `variable_scope`.
      Returns:
        A tensor representing the output of the operation.
      Raises:
        ValueError: If `data_format` is invalid.
        ValueError: Both 'rate' and `stride` are not uniformly 1.
        ValueError: If 'groups'<1.
      """
    # if no group size specified or less than/equal to zero return a normal convolution
    if groups == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding=padding,
                           activation_fn=activation_fn,
                           normalizer_fn=normalizer_fn,
                           normalizer_params=normalizer_params,
                           weights_initializer=weights_initializer,
                           weights_regularizer=weights_regularizer,
                           biases_initializer=biases_initializer,
                           biases_regularizer=biases_regularizer,
                           reuse=reuse,
                           trainable=trainable,
                           scope=scope)
    if groups < 1:
        raise ValueError("Specify a number of groups greater than zero, groups given is {}".format(groups))

    input_channels = inputs.get_shape().as_list()[-1]

    # check if the number of groups and corresponding group_size is an integer division of the input and output channels
    lowest_channels = min(input_channels, num_outputs)
    assert lowest_channels % groups == 0, "the remainder of min(input_channels,output_channels)/groups should be zero"
    assert max(input_channels,
               num_outputs) % groups == 0, "the remainder of max(input_channels,output_channels)/groups=({}) " \
                                               "should be zero".format(
        groups)

    with tf.variable_scope(scope, 'Group_Conv', [inputs], reuse=reuse) as sc:
        # define weight shape
        if isinstance(kernel_size, collections.Iterable):
            weights_shape = list(kernel_size) + [input_channels/groups] + [num_outputs]
        else:
            weights_shape = [kernel_size, kernel_size, input_channels/groups, num_outputs]

        # create weights variable
        weights = slim.variable('weights',
                                shape=weights_shape,
                                initializer=weights_initializer,
                                regularizer=weights_regularizer,
                                trainable=trainable)
        strides = [stride, stride]
        dilation_rate = [rate, rate]
        # perform grouped convolution
        outputs = grouped_convolution2D(inputs, weights, padding, groups,
                                        strides=strides,
                                        dilation_rate=dilation_rate)
        if biases_initializer is not None:
            biases = slim.variable('biases',
                                   shape=[num_outputs],
                                   initializer=biases_initializer,
                                   regularizer=biases_regularizer,
                                   trainable=trainable)
            outputs = tf.nn.bias_add(outputs, biases)
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
