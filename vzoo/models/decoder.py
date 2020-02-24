import numpy as np
import tensorflow as tf
from vzoo.core.layers import _dense, _deconv2d_bn


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/layers/utils.py
def deconv_output_dim(input_dim,
                      kernel_size,
                      strides,
                      padding='valid'):
    """Determines output length of a transposed convolution given input length.

    Parameters
    ----------
    input_dim: int
        Input length of conv (assumes symmetry)
    kernel_size : int, tuple
         Size of the convolving kernel.
    strides : int, tuple (default=1)
        Number of stridee.
    padding : str (default='same')
        Padding mode.

    Returns
    -------
    output_dim : int
        Output length of Conv2DTranspose
    """

    if not isinstance(kernel_size, int):
        kernel_size = kernel_size[0]
    if not isinstance(strides, int):
        strides = strides[0]

    if input_dim is None:
        return None
    output_dim = input_dim * strides
    if padding == 'valid':
        output_dim += max(kernel_size - strides, 0)
    elif padding == 'full':
        output_dim -= (strides + kernel_size - 2)
    return output_dim


def _conv_config_shape(layers):
    """
    Creates a list of conv layer configurations and corresponding output
    shapes given a list of model layers.
    Wild bugs may appear.
    """
    conv_configs = []
    conv_shapes = []
    config = None

    for layer in layers:
        if isinstance(layer, tf.keras.layers.Dense):
            # break loop once we've reached fully connected layers
            break

        elif isinstance(layer, tf.keras.layers.Conv2D):
            # Wait until the next conv is called to add config this allow us to
            # to update the confi with activation/bn called seperately
            if config is not None:
                # force first conv layer to have linear output
                if len(conv_configs) == 0:
                    config['activation'] = None
                conv_configs.append(config)
                conv_shapes.append(output_shape)
                config = None

            idx = len(conv_configs) + 1
            config = layer.get_config()

            input_shape = list(layer.input.shape[1:])
            output_shape = list(layer.output.shape[1:])

            strides = config['strides']
            kernel_size = config['kernel_size']
            padding = config['padding']

            output_dim = deconv_output_dim(output_shape[0],
                                           kernel_size,
                                           strides,
                                           padding)

            # add case for 'valid' padding
            if padding == 'same':
                off_dim = np.abs(input_shape[0] - output_dim)
                if off_dim > 0:
                    config['output_padding'] = strides - off_dim - 1

            config['filters'] = input_shape[-1] # channels last
            config['name'] = f'dec_conv2d_transpose_{idx}'

        elif isinstance(layer, tf.keras.layers.Activation):
            if config is not None:
                config['activation'] = layer.activation.__name__

        elif hasattr(layer, 'node_def'):
            if hasattr(layer.node_def, 'op'):
                if config is not None:
                    op = layer.node_def.op
                    # LeakyRelu -> leaky_relu
                    a = [c if c.islower() else f'_{c.lower()}' for c in op]
                    config['activation'] = ''.join(a).lstrip('_')

        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            if config is not None:
                conv_configs.append(config)
                config_shapes.append(output_shape)
                config = None

            idx = len(conv_configs) + 1
            config = layer.get_config()
            input_shape = layer.output.shape[1:]
            output_shape = layer.input.shape[1:]

            filters = input_shape[-1]
            strides = config['strides']
            kernel_size = config['pool_size']

            new_dim = deconv_output_dim(input_shape[0],
                                        kernel_size,
                                        strides)
            output_padding = max(output_shape[0] - new_dim, 0)

            usample = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                use_bias=False,
                output_padding=output_padding,
                name=f'dec_upsample_{idx}'
            )
            conv_configs.append(usample.get_config())
            conv_shapes.append(list(output_shape))
    # add final conv layer.
    if config is not None:
        conv_configs.append(config)
        conv_shapes.append(output_shape)

    return list(reversed(conv_configs)), list(reversed(conv_shapes))


def auto_decode(model, latent_tensor, output_shape):
    """Automaticly builds the core layers of a decoder given the encoder.

    The list returned stores a dictionary of layer configs to be passed in to
    transposed conv layers. Any max pooling layers in the encoder will be
    replaced with deconv layers as well.

    Parameters
    ----------
    model : tf.keras.Model
        Encoder to tranpose.
    latent_tensor : tf.Tensor
        Latent tensor of shape (batch_size, latent_dim) used as input to
        the decoder.
    output_shape : tuple
        Shape of the input to reconstruct.

    Returns
    -------
    x : tf.Tensor
        Output of decoder with the same shape as the input for the encoder.
    """
    conv_configs, conv_shapes = _conv_config_shape(model.layers)

    final_conv_shape = conv_shapes[0]
    x = _dense(latent_tensor, 256)
    x = _dense(x, np.prod(final_conv_shape))
    x = tf.reshape(x, [-1] + final_conv_shape)
    for config in conv_configs:
        x = _deconv2d_bn(x, **config)

    return x
