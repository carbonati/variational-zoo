import tensorflow as tf
from vzoo.core.layers import _conv2d_bn, _deconv2d_bn, _dense


def conv_encoder(input_tensor,
                 layer_prefix='enc_'):
    """Convolutional network for a Gaussian encoder as proposed in section A.1
    of "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
    Framework" (https://openreview.net/references/pdf?id=Sy2fzU9gl).

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor to encode.
    layer_prefix : str (default='enc_')
        Prefix of each layer name.

    Returns
    -------
    x : tf.Tensor
        Output tensor of the encoder.
    """
    x = _conv2d_bn(input_tensor,
                   filters=32,
                   kernel_size=4,
                   strides=2,
                   name=f'{layer_prefix}conv2d_1')
    x = _conv2d_bn(x,
                   filters=32,
                   kernel_size=4,
                   strides=2,
                   name=f'{layer_prefix}conv2d_2')
    x = _conv2d_bn(x,
                   filters=64,
                   kernel_size=4,
                   strides=2,
                   name=f'{layer_prefix}conv2d_3')
    x = _conv2d_bn(x,
                   filters=64,
                   kernel_size=4,
                   strides=2,
                   name=f'{layer_prefix}conv2d_4')

    x = tf.keras.layers.Flatten(name=f'{layer_prefix}flatten')(x)
    x = _dense(x,
               out_units=256,
               activation='relu',
               use_bias=False,
               name=f'{layer_prefix}fc')
    return x

def conv_decoder(latent_tensor,
                 output_shape,
                 layer_prefix='dec_'):
    """Convolutional network for a Bernoulli decoder as proposed in section A.1
    of "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
    Framework" (https://openreview.net/references/pdf?id=Sy2fzU9gl).

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor to encode.
    output_shape : tuple
        Shape of input to reconstruct.
    layer_prefix : str (default='dec_')
        Prefix of each layer name.

    Returns
    -------
    output : tf.Tensor
        Output tensor of the conv decoder of shape `output_shape`.
    """
    x = tf.reshape(latent_tensor,
                   shape=[-1, np.prod(latent_tensor.shape[1:])],
                   name=f'{layer_prefix}flatten_in')
    x = _dense(x,
               256,
               activation='relu',
               use_bias=False,
               name=f'{layer_prefix}fc_1')
    x = _deconv2d(input_tensor,
                  filters=64,
                  kernel_size=4,
                  strides=2,
                  name=f'{layer_prefix}deconv2d_1')
    x = _deconv2d(x,
                  filters=64,
                  kernel_size=4,
                  strides=2,
                  name=f'{layer_prefix}conv2d_2')
    x = _deconv2d(x,
                  filters=32,
                  kernel_size=4,
                  strides=2,
                  name=f'{layer_prefix}conv2d_3')
    x = _deconv2d(x,
                  filters=32,
                  kernel_size=4,
                  strides=2,
                  name=f'{layer_prefix}conv2d_4')
    output = tf.reshape(x, shape=[-1] + list(output_shape))
    return x

def fc_discriminator(latent_tensor,
                     hidden_units=1000,
                     activation='leaky_relu',
                     use_bias=False,
                     layer_prefix='disc_'):

    """Fully connected discriminator as proposed in "Disentangling
    by Factorising" (https://arxiv.org/pdf/1802.05983.pdf).

    The objective of the discriminator is to determine if the input is sampled
    from the true distribtuion q(z) or from a randomly shuffled distribution
    q~(z).

    Parameters
    ----------
    latent_tensor : tf.Tensor
        Tensor from the sampled latent space q(z|x).
    hidden_units : int (default=1000)
        Number of hidden units in each FC layer.
    activation : str, tf.keras.layers.Activation, tf.nn (default='leaky_relu')
        Activation function applied to output.
    use_bias : bool (default=False)
        Boolean whether the layer uses the intercept term.
    layer_prefix : str (default='disc_')
        Prefix of each layer name.

    Returns
    -------
    logits : tf.Tensor
        Logit outputs from the discriminator of size (batch_size, 2).
    probs : tf.Tensor
        Probability outputs from the discriminiator of size (bathc_size, 2).
    """
    if hasattr(tf.nn, activation):
        activation = getattr(tf.nn, activation)

    x = tf.reshape(latent_tensor,
                   shape=[-1, np.prod(latent_tensor.shape[1:])])
    x = _dense(x,
               hidden_units,
               activation,
               use_bias=use_bias,
               name=f'{layer_prefix}fc_1')
    x = _dense(x,
               hidden_units,
               activation,
               use_bias=use_bias,
               name=f'{layer_prefix}fc_2')
    x = _dense(x,
               hidden_units,
               activation,
               use_bias=use_bias,
               name=f'{layer_prefix}fc_3')
    x = _dense(x,
               hidden_units,
               activation,
               use_bias=use_bias,
               name=f'{layer_prefix}fc_4')
    x = _dense(x,
               hidden_units,
               activation,
               use_bias=use_bias,
               name=f'{layer_prefix}fc_5')
    x = _dense(x,
               hidden_units,
               activation,
               use_bias=use_bias,
               name=f'{layer_prefix}fc_6')
    logits = _dense(x,
                    out_units=2,
                    activation=None,
                    use_bias=False,
                    name=f'{layer_prefix}logit')
    probs = tf.nn.softmax(logits, name=f'{layer_prefix}probs')
    # without clipping the discriminator diverges (5 hours of debuging)
    probs = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
    return logits, probs

def fc_encoder(input_tensor,
               hidden_units=1200,
               activation='relu',
               use_bias=False,
               layer_prefix='enc_'):
    """MLP encoder as proposed in section A.4 of "beta-VAE: Learning Basic
    Visual Concepts with a Constrained Variational Framework"
    (https://openreview.net/references/pdf?id=Sy2fzU9gl).

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor to encode.
    hidden_units : int (default=1200)
        Number of hidden units in each FC layer.
    activation : str, tf.keras.layers.Activation, tf.nn (default='relu')
        Activation function applied to linear dense layer.
    use_bias : bool (default=False)
        Boolean whether the layer uses the intercept term.
    layer_prefix : str (default='enc_')
        Prefix of each layer name.

    Returns
    -------
    x : tf.Tensor
        Output tensor of the fully connected encoder.
    """
    x = tf.reshape(latent_tensor,
                   shape=[-1, np.prod(input_tensor.shape[1:])],
                   name='flatten')
    x = _dense(x,
               hidden_units,
               activation,
               use_bias=use_bias,
               name=f'{layer_prefix}fc_1')
    x = _dense(x,
               hidden_units,
               activation,
               use_bias=use_bias,
               name=f'{layer_prefix}fc_2')
    return x

def fc_decoder(latent_tensor,
               output_shape,
               hidden_units=1200,
               activation='tanh',
               use_bias=False,
               layer_prefix='dec_'):
    """MLP decoder as proposed in section A.4 of "beta-VAE: Learning Basic
    Visual Concepts with a Constrained Variational Framework"
    (https://openreview.net/references/pdf?id=Sy2fzU9gl).

    Parameters
    ----------
    latent_tensor : tf.Tensor
        Input tensor to encode.
    output_shape : tuple
        Shape of input to reconstruct.
    hidden_units : int (default=1200)
        Number of hidden units in each FC layer.
    activation : str, tf.keras.layers.Activation, tf.nn (default='tanh')
        Activation function applied to linear dense layer.
    use_bias : bool (default=False)
        Boolean whether the layer uses the intercept term.
    layer_prefix : str (default='dec_')
        Prefix of each layer name.

    Returns
    -------
    output : tf.Tensor
        Reconstructed output of size `output_shape`
    """
    x = tf.reshape(x,
                   shape=[-1, np.prod(x.shape[1:])])
    x = _dense(x,
               hidden_units,
               activation,
               use_bias=use_bias,
               name=f'{layer_prefix}fc_1')
    x = _dense(x,
               hidden_units,
               activation,
               use_bias=use_bias,
               name=f'{layer_prefix}fc_2')
    x = _dense(x,
               np.prod(output_shape),
               activation=None,
               use_bias=use_bias,
               name=f'{layer_prefix}output')
    output = tf.reshape(x, shape=[-1] + list(output_shape))
    return output
