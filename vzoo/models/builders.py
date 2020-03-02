import tensorflow as tf
from vzoo.core.layers import _dense
from vzoo.models.decoder import auto_decode


def build_inference_model(input_tensor,
                          encoder_fn,
                          latent_dim,
                          name='encoder',
                          **kwargs):
    """Builds a Gaussian inference model (encoder), which takes in an
    `input_tensor`, applies an `encoder_fn`, and returns the posterior q(z|x),
    which is parameterized using a mean and log variance FC layer.

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor to encode.
    encoder_fn : function
        Encoder function that expects an input tensor `x` and returns
        the output tensor of encoder that will be connected to a mean and
        log variance dense layers parameterizing the latent space. Example:
            input_tensor = tf.keras.layers.Input((64, 64, 3))
            x = encoder_fn(input_tensor)
            z_mean = tf.keras.layers.Dense(latent_dim)(x)
            z_log_var = tf.keras.layers.Dense(latent_dim)(x)
    latent_dim : int
        Dimension of latent space.
    name : str (default='encoder')
        Name of inference model.

    Returns
    -------
    model : tf.keras.Model
        Gaussian encoder which accepts `input_tensor` and outputs two tensors:
            z_mean : Mean latent vector of the encoder.
            z_log_var : Log variance latent vector of the encoder.
    """
    x = encoder_fn(input_tensor, **kwargs)
    if len(x.shape) > 2:
        x = tf.keras.layers.Flatten()(x)

    # parameterize the latent space with a mean a log variance FC layer.
    z_mean = _dense(x,
                    latent_dim,
                    activation=None,
                    use_bias=False,
                    name='enc_mean')
    z_log_var = _dense(x,
                       latent_dim,
                       activation=None,
                       use_bias=False,
                       name='enc_logvar')

    model = tf.keras.models.Model(
        input_tensor,
        [z_mean, z_log_var],
        name=name
    )
    return model


def build_generative_model(latent_dim,
                           output_shape,
                           generator_fn=None,
                           inference_model=None,
                           name=None):
    """Builds generative model (decoder), by passing in a latent tensor p(z|x),
    applies a `generator_fn`, and returns the reconstruction p(x|z).

    Parameters
    ----------
    latent_dim : int
        Dimension of latent space.
    output_shape : tuple/list
        Shape of output to reconstruct.
    generator_fn : function (defualt=None)
        Generator/decoder function that expects an input tensor `x` and
        `output_tensor` to return the reconstructed tensor of an observation.
        Example:
            latent_tensor = tf.keras.layers.Input(latent_dim)
            x_decoded = generator_fn(latent_tensor, output_shape)
    inference_model : tf.keras.Model (default=None)
        Encoder model that will be automatically transposed if provided.
    name : str (default=None)
        Name of model.

    Returns
    -------
    model : tf.keras.Model
        Decoder which accepts a tensor of size `latent_dim` and outputs the
        reconstructed input.
    """
    latent_tensor = tf.keras.layers.Input((latent_dim))

    if generator_fn:
        x = generator_fn(latent_tensor, output_shape)
    elif inference_model is not None:
        x = auto_decode(inference_model, latent_tensor, output_shape)
    else:
        raise ValueError("'generator_fn' and 'inference_model' cannot both be None")

    model = tf.keras.models.Model(latent_tensor, x, name=name)
    return model


def build_discriminator(input_tensor,
                       discriminator_fn,
                       name='discriminator',
                       **kwargs):
    """Builds a discriminator that determines if the input is from the
    data distribution, P_data, or model distribution, P_G.

    Builds a discriminator that will be called twice. Once with inputs
    from the true distribution q(z|x) and from a shuffled distribution q~(z|x).
    The objective of the discriminator is to predict if the inputs is from
    q(z|x) or q~(z|x).

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor to discriminate of shape (batch_size, *input_shape).
    discriminator_fn : function
        Discriminator function that expects an input tensor `input_tensor` with
        any other parameters passed as `**kwargs` and returns an output tensor(s).

    Returns
    -------
    discriminator : tf.keras.Model
        Discriminator model that accepts an `input_tensor` of shape
        (batch_size, *input_shape) and returns an output tensor(s).
    """
    output = discriminator_fn(input_tensor, **kwargs)
    discriminator = tf.keras.Model(input_tensor, output, name=name)
    return discriminator
