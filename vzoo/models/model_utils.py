import os
import tensorflow as tf
from vzoo import models


def permute_dims(z):
    """Algorithm 1 of "Disentangling by Factorising"
    (https://arxiv.org/pdf/1802.05983.pdf).
    """
    # https://github.com/tensorflow/tensorflow/issues/6269#issuecomment-465850464
    batch_size, latent_dim = z.shape
    z_perm = []
    for i in range(latent_dim):
        z_perm.append(
            tf.gather(z[:,i], tf.random.shuffle(tf.range(batch_size)))
        )
    return tf.stack(z_perm, axis=1, name="z_permuted")


def save_model(model, model_dir, verbose=0):
    """Saves model weights and configurations to disk."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if verbose:
        print(f'Saving model to {model_dir}')
    model.save(model_dir)


def load_model(model_dir):
    """
    Loads model weights and configurations, and instantiates a VAEZOO class.
    """
    _model = tf.keras.models.load_model(model_dir)
    if hasattr(_model, 'vae'):
        encoder = _model.inference_model
        decoder = _model.generative_model
        sampling_fn = _model.sampling_fn
    else:
        encoder = _model.get_layer('encoder')
        decoder = _model.get_layer('decoder')
        sampling_fn = _model.get_layer('sampling_fn')

    input_shape = encoder.input.shape[1:].as_list()
    latent_dim = decoder.input.shape[1]
    input_tensor = tf.keras.layers.Input(input_shape,
                                         name='enc_input')

    if hasattr(_model, 'discriminator'):
        model = models.vae.FactorVAE(input_shape=input_shape,
                                     latent_dim=latent_dim)
        model.discriminator = _model.discriminator
    else:
        model = models.vae.VAE(input_shape=input_shape,
                               latent_dim=latent_dim)

    model.inference_model = encoder
    model.generative_model = decoder
    model.sampling_fn = sampling_fn
    model._set_inputs(input_tensor)

    return model


def check_loss_fn(loss_fn):
    """Checks if `loss_fn` is a recognized str or custom function."""
    if isinstance(loss_fn, str):
        if loss_fn in config.LOSS_FN_DICT.keys():
           loss_fn = LOSS_FN_DICT[loss_fn]
        else:
            raise ValueError("Unrecognized 'loss_fn' {}.".format(loss_fn),
                             "Expected a function or one of {}".format(", ".join(list(LOSS_FN_DICT.keys()))))
    return loss_fn
