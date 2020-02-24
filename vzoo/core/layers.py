import tensorflow as tf
from functools import partial


class GaussianSampler(tf.keras.layers.Layer):
    """Samples from a Gaussian using the "reparamerization trick" defined by
    the mean and log variance for a latent variable as shown in "Auto-Encoding
    Variational Bayes" (https://arxiv.org/pdf/1312.6114.pdf).

    z_i = mu_i + sigma_i * eps
    s.t. eps denotes gaussian noise. This equation becomes
        z_i = mu_i + exp(z_log_var_i / 2) * eps
    since z_sigma = exp(z_log_var / 2).

    Parameters
    ----------
    mean : float (default=0)
        Mean of a gaussian to sample noise from.
    stddev : float (default=1)
        Standard deviatioin of gaussian to sample noise from.
    """
    def __init__(self,
                 mean=0,
                 stddev=1,
                 name='sampling_fn',
                 **kwargs):
        super(GaussianSampler, self).__init__(name=name, **kwargs)
        self.mean = mean
        self.stddev = stddev

    def call(self, inputs):
        z_mean, z_log_var = inputs
        shape = z_mean.shape if z_mean.shape[0] else z_mean.shape[1:]
        eps = tf.random.normal(shape=shape,
                               mean=self.mean,
                               stddev=self.stddev)
        return z_mean + tf.exp(z_log_var / 2.) * eps


    def get_config(self):
        config = super(GaussianSampler, self).get_config()
        config.update({
            'mean': self.mean,
            'stddev': self.stddev,
        })
        return config


def _conv2d_bn(x,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               activation='relu',
               use_bias=False,
               bn_before=None,
               name=None,
               **kwargs):
    """Wrapper to construct a 2D conv + BN layer.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    filters : int
        Number of output filters in the Conv2D.
    kernel_size : int, tuple
         Size of the convolving kernel.
    strides : int (default=1)
        Number of stridee in the Conv2D.
    padding : str (default='same')
        Padding mode in Conv2D
    activation : str, tf.keras.layers.Activation, tf.nn (default='relu')
        Activation function applied to output.
    use_bias : bool (default=False)
        Boolean whether the layer uses the intercept term.
    bn_before : bool, None (defualt=None)
        Whether to place a batch norm layer before or after activation,
        or not at all.
            True - batch norm before activation.
            False - batch norm after activation.
            None - No batch norm.
    name : str (default=None)
        Name/prefix of any conv2d, activation, batch_norm ops called.

    Returns
    -------
        x : tf.Tensor
            Output tensor after applying a conv2d (+ batch norm).
    """

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name,
        **kwargs
    )(x)

    if activation is None:
        activation = 'linear'
    if isinstance(activation, str):
        act_name = None if name is None else f'{name}_{activation}'
        act = tf.keras.layers.Activation(activation, name=act_name)
    else: # tf.nn
        act_name = None if name is None else f'{name}_{activation.__name__}'
        act = partial(activation, name=act_name)

    bn_name = None if name is None else f'{name}_bn'

    # apply bn before, after, or not at all.
    if bn_before is True:
        x = tf.keras.layers.BatchNormalization(name=bn_name)(x)
        x = act(x)
    elif bn_before is False:
        x = act(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name)(x)
    else:
        x = act(x)

    return x


def _deconv2d_bn(x,
                filters,
                kernel_size,
                strides=1,
                padding='same',
                activation='tanh',
                bn_before=None,
                use_bias=False,
                name=None,
                **kwargs):
    """Wrapper to construct a tranposed 2D conv (deconv) + BN layer.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    filters : int, tuple
        Number of output filters in the Conv2DTranpose.
    kernel_size : int, tuple
         Size of the convolving kernel.
    strides : int, tuple (default=1)
        Number of stridee in the Conv2DTranpose.
    padding : str (default='same')
        Padding mode in Conv2DTranpose.
    activation : str, tf.keras.layers.Activation, tf.nn (default='relu')
        Activation function applied to output.
    use_bias : bool (default=False)
        Boolean whether the layer uses the intercept term.
    bn_before : bool, None (defualt=None)
        Whether to place a batch norm layer before or after activation,
        or not at all.
            True - batch norm before activation.
            False - batch norm after activation.
            None - No batch norm.
    name : str (default=None)
        Name/prefix of any conv2dtranpose, activation, batch_norm ops called.

    Returns
    -------
        x : tf.Tensor
            Output tensor after applying a conv2dtranpose (+ batch norm).
    """
    x = tf.keras.layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name,
        **kwargs
    )(x)

    if activation is None:
        activation = 'linear'
    if isinstance(activation, str):
        act_name = None if name is None else f'{name}_{activation}'
        act = tf.keras.layers.Activation(activation, name=act_name)
    else:
        act_name = None if name is None else f'{name}_{activation.__name__}'
        act = partial(activation, name=act_name)
    bn_name = None if name is None else f'{name}_bn'

    if bn_before is True:
        x = tf.keras.layers.BatchNormalization(name=bn_name)(x)
        x = act(x)
    elif bn_before is False:
        x = act(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name)(x)
    else:
        x = act(x)

    return x


def _dense(x,
           out_units,
           activation='relu',
           use_bias=False,
           drop_rate=0,
           name=None,
           **kwargs):
    """Wrapper to construct a fully connected layer.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    out_units : int
        Size of output space.
    activation : str, tf.keras.layers.Activation, tf.nn (default='relu')
        String or activation function.
    use_bias : bool (default=False)
        Boolean whether the layer uses the intercept term.
    drop_rate : float (default=0)
        Dropout rate.
    name : str (default=None)
        Name/prefix of any dense, activation, dropout ops used.

    Returns
    -------
        x : tf.Tensor
            Output tensor after applying a dense layer.
    """
    x = tf.keras.layers.Dense(out_units,
                              use_bias=use_bias,
                              name=name,
                              **kwargs)(x)

    if activation is None:
        activation = 'linear'
    if isinstance(activation, str):
        act_name = None if name is None else f'{name}_{activation}'
        act = tf.keras.layers.Activation(activation, name=act_name)
    else:
        act_name = None if name is None else f'{name}_{activation.__name__}'
        act = partial(activation, name=act_name)

    x = act(x)
    if drop_rate > 0:
        drop_name = None if name is None else f'{name}_dropout'
        x = tf.keras.layers.Dropout(drop_rate, name=drop_rate)(x)

    return x

