import tensorflow as tf
from vzoo.core.networks import conv_encoder
from vzoo.core.layers import GaussianSampler
from vzoo.models.builders import (build_inference_model,
                                  build_generative_model,
                                  build_discriminator_model)
from vzoo.models.model_utils import permute_dims


class BaseVAE(tf.keras.Model):
    """Base class for Variational Autoencoder."""

    def __init__(self,
                 input_shape,
                 latent_dim,
                 encoder_fn,
                 decoder_fn,
                 sampling_fn):
        super(BaseVAE, self).__init__()

        self._input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_fn = encoder_fn
        self.decoder_fn = decoder_fn
        self.sampling_fn = sampling_fn

        self.input_tensor = None
        self.inference_model = None
        self.generative_model = None
        self.vae = None

        self._check_args()

    @property
    def input_shape(self):
        """Returns the encoders input shape."""
        return self._input_shape

    @property
    def output_shape(self):
        """Returns the decoders output shape."""
        return self._input_shape

    def _check_args(self):
        """Sets default `encoder_fn` and `sampling_fn` if none specified."""
        self.sampling_fn = self.sampling_fn or GaussianSampler
        self.encoder_fn = self.encoder_fn or conv_encoder
        self._init_args()

    def _init_args(self):
        """Instantiates layers required to before training."""
        self.sampling_fn = self.sampling_fn(name='sampling_fn')

    def _build_model(self, input_shape):
        self.build_model(input_shape)

    def build_model(self, input_shape):
        """Logic to build the encoder, decoder, and VAE lives here.

        This can/should be called during instantiation.
        Equivalent to `compile`.

        Parameters
        ----------
        input_shape : list, tuple
            Shape of input tensor -> self.input_shape
        """
        raise NotImplementedError

    def encode(self, x):
        """Encodes an input to a mean and log variance latent representation.

        Parameters
        ----------
        x : tf.Tensor, np.ndarray
            Input tensor of shape `self.input_shape`.

        Returns
        -------
        z_mean : tf.Tensor
            Mean latent vector of the encoder.
        z_log_var : tf.Tensor
            Log variance latent vector of the encoder.
        """
        raise NotImplementedError

    def reparameterize(self, z_mean, z_log_var):
        """Samples a latent tensor from q(z|x) using the reparamterization trick
        with `self.sample_fn`.

        This is required to propogate gradients from the decoder to encoder.

        Parameters
        ----------
        z_mean : tf.Tensor
            Mean latent vector of the encoder.
        z_log_var : tf.Tensor
            Log variance latent vector of the encoder.

        Returns
        -------
        z_samp : tf.Tensor
            Output tensor from `self.sample_fn`.
        """
        raise NotImplementedError

    def decode(self, z):
        """Generates a reconstructed observation from a latent tensor.

        Parameters
        ----------
        z : tf.Tensor
            Latent tensor sampled from q(z|x).

        Returns
        -------
        x_decoded : tf.Tensor
            Obsrevation generated from the latent space.
        """
        raise NotImplementedError

    def call(self, inputs, training=None):
        """Code to perform a forward pass in the network goes here.

        Parameters
        ----------
        inputs : tf.Tensor, np.ndarray
            Input tensor of shape `self.input_shape`.
        training : bool (default=None)
            True : training mode.
            False : inference mode.
            None : keras learning phase.

        Returns
        -------
        x_decoded : tf.Tensor, np.ndarray
            Reconstruction of `inputs`.
        """
        raise NotImplementedError

    def sample_z(self, x):
        """Given an input samples a latent representation from q(z|x).

        Parameters
        ----------
        x : tf.Tensor, np.ndarray
            Input tensor of shape `self.input_shape`.

        Returns
        -------
        z_samp : tf.Tensor, np.ndarray
            Output tensor from `self.sample_fn`.
        """
        raise NotImplementedError


class VAE(BaseVAE):
    """Variation Autoencoder as originally proposed in "Auto-Encoding
    Variational Bayes" (https://arxiv.org/pdf/1312.6114.pdf).

    This class is sufficient for training many variants, which include
    beta-VAE, TCVAE, DIP-VAE, etc. The loss functions for all models can be
    found in losses.py where each function includes a reference to the
    respective paper.

    Parameters
    ----------
    input_shape : tuple, list
        Shape of input to encode and decode.
    latent_dim : int
        Dimension of latent space.
    encoder_fn : function (default=None)
        Encoder function that expects an input tensor `x` and returns
        the output tensor of encoder that will be connected to a mean and
        log variance dense layers parameterizing the latent space. Example:
            input_tensor = tf.keras.layers.Input((64, 64, 3))
            x = encoder_fn(input_tensor)
            z_mean = tf.keras.layers.Dense(latent_dim)(x)
            z_log_var = tf.keras.layers.Dense(latent_dim)(x)
        By default if no `encoder_fn` is passed in `conv_encoder`, which uses
        the proposed architecture in "beta-VAE: Learning Basic Visual Concepts
        with a Constrained Variational Framework"
        (https://openreview.net/pdf?id=Sy2fzU9gl).
    decoder_fn : function (defualt=None)
        Decoder function that expects an input tensor `x` and `output_tensor`
        and returns the reconstructed tensor of an observation. Example:
            latent_tensor = tf.keras.layers.Input(latent_dim)
            x_decoded = decoder_fn(latent_tensor, output_shape)
            loss = compute_loss(x, x_decoded)
        By default if no `decoder_fn` is passed in the `encoder_fn` will
        automatically be transposed.
    sampling_fn : tf.keras.layers.Layer (default=GaussianSampler)
        Layer that accepts a list of inputs [z_mean, z_log_var] and returns
        the sampled latent representation using the reparametrization trick
        shown in "Auto-Encoding Variational Bayes"
        (https://arxiv.org/pdf/1312.6114.pdf).
            z_samp = sampling_fn([z_mean, z_log_var])

    Example
    -------
    # beta-VAE proposed in https://openreview.net/pdf?id=Sy2fzU9gl
    beta = 5
    model = VAE(input_shape=(64, 64, 1),
                latent_dim=10,
                encoder_fn=conv_encoder)
    z_mean, z_log_var = model.encode(x)
    z_samp = model.reparameterize(z_mean, z_log_var)
    x_decoded = model.decode(z_samp)

    recon_loss = bernoulli_loss(x, x_decoded, from_logits=True)
    kl_loss = compute_gaussian_kl_loss(z_mean, z_log_var)
    elbo_loss = recon_loss + beta * kl_loss
    """
    def __init__(self,
                 input_shape,
                 latent_dim,
                 encoder_fn=None,
                 decoder_fn=None,
                 sampling_fn=None,
                 **kwargs):
        super(VAE, self).__init__(input_shape,
                                  latent_dim,
                                  encoder_fn,
                                  decoder_fn,
                                  sampling_fn)

    def build_model(self):
        self.input_tensor = tf.keras.layers.Input(
            self.input_shape,
            name='enc_input'
        )
        self.inference_model = build_inference_model(
            self.input_tensor,
            self.encoder_fn,
            latent_dim=self.latent_dim
        )
        self.generative_model = build_generative_model(
            self.latent_dim,
            output_shape=self.input_shape,
            decoder_fn=self.decoder_fn,
            inference_model=self.inference_model
        )
        x_decoded = self.call(self.input_tensor)
        self.vae = tf.keras.Model(self.input_tensor, x_decoded, name='vae')
        self._set_inputs(self.input_tensor)

    def encode(self, x):
        return self.inference_model(x)

    def reparameterize(self, z_mean, z_log_var):
        return self.sampling_fn([z_mean, z_log_var])

    def decode(self, z):
        return self.generative_model(z)

    def sample_z(self, x):
        z_mean, z_log_var = self.encode(x)
        return self.reparameterize(z_mean, z_log_var)

    def call(self, inputs, training=None):
        if len(inputs.shape) == len(self.input_shape):
            # add extra batch dimension
            inputs = tf.expand_dims(inputs, axis=0)
        z_mean, z_log_var = self.encode(inputs)
        z_samp = self.reparameterize(z_mean, z_log_var)
        return self.decode(z_samp)


class FactorVAE(VAE):
    """FactorVAE proposed in "Disentangling by Factorizing"
    (https://arxiv.org/pdf/1802.05983).

    This class inherits VAE with the addition of building a discriminator
    that should be called twice. Once with inputs from the true distribution
    q(z|x) and a second from a shuffled distribution q~(z|x). The objective of
    the discriminator is to predict if the inputs is from q(z|x) or q~(z|x).

    factorised = have diagonal covariance matrix

    Parameters
    ----------
    input_shape : tuple, list
        Shape of input to encode and decode.
    latent_dim : int
        Dimension of latent space.
    discriminator_fn : function
        Discriminator function that expects an input tensor of shape
        (batch_size, latent_dim) and returns a logit and probability vector to
        compute the FactorVAE loss. Example:
            latent_tensor = tf.keras.layers.Input(latent_dim)
            logits, probs = fc_discriminiator(latent_tensor)
        By default if `discriminator_fn` left as None, `fc_discriminator`,
        which uses the proposed architecture in the paper will be used.

    Example
    -------
    # Algorithm 2 FactorVAE proposed in the original paper.
    gamma = 10
    model = FactorVAE(input_shape=(64, 64, 1),
                      latent_dim=10,
                      encoder_fn=conv_encoder,
                      discriminator_fn=fc_discriminiator)

    z_mean, z_logvar = model.encode(x)
    z_samp = model.reparameterize(z_mean, z_logvar)
    x_decoded = model.decode(z_samp)

    z_perm = permute_dims(z_samp)
    z_logits, z_probs = model.discriminator(z_samp)
    z_perm_logits, z_perm_probs = model.discriminator(z_perm)

    recon_loss = bernoulli_loss(x, x_decoded, from_logits=True)
    kl_loss = compute_gaussian_kl_loss(z_mean, z_logvar)
    elbo_loss = recon_loss + beta * kl_loss

    tc_loss = tf.reduce_mean(z_logits[:,0] - z_logits[:,1])
    factor_vae_loss = elbo_loss + gamma * tc_loss
    disc_loss = - 0.5 * (  tf.reduce_mean(tf.math.log(z_probs[:, 0])),
                         + tf.reduce_mean(tf.math.log(z_perm_probs[:, 1]))  )
    """
    def __init__(self,
                 input_shape,
                 latent_dim,
                 discriminator_fn=None,
                 **kwargs):
        super(FactorVAE, self).__init__(input_shape,
                                        latent_dim,
                                        **kwargs)
        self.discriminator_fn = discriminator_fn or fc_discriminator
        self.discriminator = None
        self.build_model()

    def build_model(self):
        input_tensor = tf.keras.layers.Input(self._input_shape,
                                             name='enc_input')
        self.inference_model = build_inference_model(
            input_tensor,
            self.encoder_fn,
            latent_dim=self.latent_dim
        )
        self.generative_model = build_generative_model(
            self.latent_dim,
            output_shape=self._input_shape,
            decoder_fn=self.decoder_fn,
            inference_model=self.inference_model
        )

        latent_tensor = tf.keras.layers.Input((self.latent_dim),
                                              name='disc_input')
        self.discriminator = build_discriminator(
            latent_tensor,
            self.discriminator_fn
        )

        x_decoded = self.call(input_tensor)
        self.vae = tf.keras.Model(input_tensor, x_decoded, name='vae')
        self._set_inputs(input_tensor)


class BVAE(VAE):

    def __init__(self):
        pass


class TCVAE(VAE):

    def __init__(self):
        pass


class DIPVAE(VAE):

    def __init__(self):
        pass


class AnnealedVAE(VAE):

    def __init__(self):
        pass


class VQVAE(VAE):

    def __init__(self):
        pass
