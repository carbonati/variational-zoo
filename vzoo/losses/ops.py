import math
import numpy as np
import tensorflow as tf


def compute_gaussian_kl(mean, log_var):
    """
    Calculates KL divergence between the latent space q(z|x) and a
    gaussian prior p(z).

    KL(q(z|x) || p(z))
    """
    gaussian_kl = -0.5 * tf.reduce_mean(
        tf.reduce_sum(
            1. + log_var - tf.square(mean) - tf.exp(log_var),
            axis=1
        )
    )
    return gaussian_kl


def gaussian_log_density(sample, mean, log_var):
    """Computes the log density of a Gaussian."""
    log2pi = tf.math.log(2. * math.pi)
    inv_sigma = tf.exp(-log_var)
    delta = (sample - mean)
    return - 0.5 * (tf.square(delta) * inv_sigma + log_var + log2pi)


def compute_batch_gaussian_density(sample, mean, log_var):
    """Computes the gaussian log density between each sample in the batch.

    Takes in a 2D matrices of shape (batch_size, latent_dim) and returns a
    3D matrix of shape (batch_size, batch_size, latent_dim).

    Parameters
    ----------
    sample : Tf.tensor
        Sampled latent representation of shape (batch_size, latent_dim)
    mean : Tf.tensor
        Mean latent tensor of shape (batch_size, latent_dim)
    logg_var : Tf.tensor
        Log variance latent tensor of shape (batch_size, latent_dim)

    Returns
    -------
    batch_log_qz : tf.Tensor
        Gaussian log density matrix between each sample of shape
        (batch_size, batch_size, latent_dim).
    """
    batch_log_qz = gaussian_log_density(
        sample=tf.expand_dims(sample, 1),
        mean=tf.expand_dims(mean, 0),
        log_var=tf.expand_dims(log_var, 0)
    )
    return batch_log_qz


def compute_log_qz(sample, mean, log_var):
    """Computes log(q(z))"""
    log_prob_qz = compute_batch_gaussian_density(sample, mean, log_var)
    log_qz = tf.reduce_logsumexp(
        tf.reduce_sum(log_prob_qz, axis=2),
        axis=1,
    )
    return log_qz


def compute_log_prod_qz_i(sample, mean, log_var):
    log_prob_qz = compute_batch_gaussian_density(sample, mean, log_var)
    log_prod_qzi = tf.reduce_sum(
          tf.reduce_logsumexp(log_prob_qz, axis=1),
          axis=1,
    )
    return log_prod_qzi


def compute_log_qz_cond_x(sample, mean, log_var):
    log_qz_cond_x = tf.reduce_sum(
        gaussian_log_density(sample, mean, log_var),
        axis=1,
    )
    return log_qz_cond_x


def compute_log_pz(sample, mean, log_var):
    log_pz = tf.reduce_sum(
        gaussian_log_density(
            sample,
            tf.zeros(sample.shape),
            tf.zeros(sample.shape),
        ),
        axis=1,
    )
    return log_pz


def compute_cov_matrix(X):
    """Computes the covariance of a tensor."""
    X -= tf.reduce_mean(X, axis=0)
    fact = X.shape[0] - 1
    cov_X = tf.tensordot(tf.transpose(X), tf.math.conj(X), 1) / fact
    return cov_X


def compute_on_off_diag(cov_matrix):
    """Computes the on and off diagonal of a tensor."""
    diag = tf.linalg.diag_part(cov_matrix)
    off_diag = cov_matrix - tf.linalg.diag(diag)
    return diag, off_diag


def _entropy(x, base=None, axis=0, eps=1e-12):
    """Calculates entropy for a sequence of classes or probabilities."""
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    p = (x + eps) / np.sum(x + eps, axis=axis, keepdims=True)
    H = - np.sum(p * np.log(p + eps), axis=axis)
    if base is not None:
        H /= np.log(base + eps)
    return H
