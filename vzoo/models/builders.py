import tensorflow as tf
from vzoo.core.layers import _dense
from vzoo.models.decoder import auto_decode


def build_inference_model(input_tensor,
                          encoder_fn,
                          latent_dim,
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
        name='encoder'
    )
    return model

def build_generative_model(latent_dim,
                           output_shape=None,
                           decoder_fn=None,
                           inference_model=None):
    """Builds generative model (decoder), by passing in a latent tensor (size
    `latent_dim`) p(z|x), applies a `decoder_fn`, and returns the reconstruction
    p(x|z).

    Parameters
    ----------
    latent_dim :
        Dimension of latent space.
    output_shape : tuple (default=None)
        Shape of input to reconstruct.
    decoder_fn : function (defualt=None)
        Decoder function that expects an input tensor `x` and `output_tensor`
        and returns the reconstructed tensor of an observation. Example:
            latent_tensor = tf.keras.layers.Input(latent_dim)
            x_decoded = decoder_fn(latent_tensor, output_shape)
    inference_model : tf.keras.Model (default=None)
        Encoder model that will be automatically transposed if provided.

    Returns
    -------
    model : tf.keras.Model
        Decoder which accepts a tensor of size `latent_dim` and outputs the
        reconstructed input.
    """
    latent_tensor = tf.keras.layers.Input((latent_dim),
                                          name='decoder_input')
    if decoder_fn:
        x = decoder_fn(latent_tensor, output_shape)
    elif inference_model is not None:
        x = auto_decode(inference_model, latent_tensor, output_shape)
    else:
        raise ValueError("'decoder_fn' and 'inference_model' cannot both be None")

    model = tf.keras.models.Model(
        latent_tensor,
        x,
        name='decoder'
    )
    return model


def build_discriminator_model(input_tensor,
                              discriminator_fn,
                              **kwargs):
    """Builds a discriminator that will be called twice. Once with inputs
    from the true distribution q(z|x) and from a shuffled distribution q~(z|x).
    The objective of the discriminator is to predict if the inputs is from
    q(z|x) or q~(z|x).

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor to discriminate of shape (batch_size, latent_dim).
    discriminator_fn : function
        Discriminator function that expects an input tensor `input_tensor` with
        any other parameters passed as `**kwargs` and returns two output tensors:
            logits : tf.Tensor
                Logit outputs from the discriminator of size (batch_size, 2).
            probs : tf.Tensor
                Probability outputs of the `logits` of size (bathc_size, 2).
        Example:
            latent_tensor = tf.keras.layers.Input(latent_dim)
            logits, probs = fc_discriminiator(latent_tensor)
    Returns
    -------
    discriminator : tf.keras.Model
        Discriminator model that accepts an `input_tensor` of size
        (batch_size, latent_dim) and outputs two tensors:
            logits : Logit vector of th discriminator.
            probs : Probability vector of th discriminator.
    """
    logits, probs = discriminator_fn(input_tensor, **kwargs)
    discriminator = tf.keras.Model(input_tensor,
                                   [logits, probs],
                                   name='discriminator')
    return discriminator
