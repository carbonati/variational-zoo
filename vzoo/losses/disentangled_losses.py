import math
import numpy as np
import tensorflow as tf
from vzoo.losses import ops, recon_losses
from vzoo.evaluation.writer_utils import log_metrics
from vzoo.models.model_utils import permute_dims

__all__ = [
    'compute_elbo_loss',
    'compute_beta_vae_loss',
    'compute_annealed_loss',
    'compute_factor_vae_loss',
    'compute_tc_vae_loss',
    'compute_dip_vae_loss',
]


def compute_loss(model,
                 loss_fn,
                 x,
                 writer=None,
                 training=True,
                 params={}):
    """Computes and records loss for a batch."""
    losses = loss_fn(model, x, params)

    if writer is not None:
        if training:
            metric_dict = params.get('train_metrics')
        else:
            metric_dict = params.get('val_metrics')
        log_metrics(writer, losses, metric_dict)

    return losses


def compute_elbo_loss(model, x, params=None):
    """Equation 3 of "Auto-Encoding Variational Bayes"
    (https://arxiv.org/pdf/1312.6114.pdf).

    L = E[log p(x|z)] - KL(q(z|x) || p(z))
      = recon_loss - kl_loss

    Parameters
    ----------
    model : VAE
        Module that subclasses a tf.keras.Model that can encode,
        reparameterize and decode the input with.
    x : tf.Tensor
        Batch to make forward pass and calculate loss for.
    params : dict (default=None)
        Parameters/keyword arguments for computing the loss.

    Returns
    -------
    losses : dict
        recon_loss : Reconstruction loss, E[log p(x|z)]
        kl_loss : KL divergence, KL(q(z|x) || p(z))
        elbo_loss : Evidence lower bound, recon_loss + kl_loss
        loss : elbo_loss
    """
    params = params or {}
    recon_loss_fn = params.get('recon_loss_fn', recon_losses.bernoulli_loss)

    z_mean, z_logvar = model.encode(x)
    z_samp = model.reparameterize(z_mean, z_logvar)
    x_decoded = model.decode(z_samp)

    recon_loss = recon_loss_fn(x, x_decoded, from_logits=True)
    kl_loss = ops.compute_gaussian_kl(z_mean, z_logvar)
    elbo_loss = recon_loss + kl_loss

    losses = {
        'loss': elbo_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'elbo_loss': elbo_loss
    }
    return losses


def compute_beta_vae_loss(model, x, params=None):
    """Equation 4 of "beta-VAE: Learning Basic Visual Concepts with a
    Constrained Variational Framework"
    (https://openreview.net/pdf?id=Sy2fzU9gl).

    L = E[log p(x|z)] - B * KL(q(z|x) || p(z))
      = recon_loss - beta * kl_loss

    Parameters
    ----------
    model : VAE
        Module that subclasses a tf.keras.Model that can encode,
        reparameterize and decode the input.
    x : tf.Tensor
        Batch to make a forward pass with.
    params : dict (default=None)
        Parameters/keyword arguments for computing the loss.

    Returns
    -------
    losses : dict
        recon_loss : Reconstruction loss, E[log p(x|z)]
        kl_loss : KL divergence, KL(q(z|x) || p(z))
        elbo_loss : Evidence lower bound, recon_loss + beta * kl_loss
        loss : elbo_loss
    """
    beta = params.get('beta', 5)
    recon_loss_fn = params.get('recon_loss_fn', recon_losses.bernoulli_loss)

    z_mean, z_logvar = model.encode(x)
    z_samp = model.reparameterize(z_mean, z_logvar)
    x_decoded = model.decode(z_samp)

    recon_loss = recon_loss_fn(x, x_decoded, from_logits=True)
    kl_loss = ops.compute_gaussian_kl(z_mean, z_logvar)
    elbo_loss = recon_loss + beta * kl_loss

    losses = {
        'loss': elbo_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'elbo_loss': elbo_loss
    }
    return losses


def compute_annealed_loss(model, x, params={}):
    """Equation 8 of "Understanding disentangling in β-VAE"
    (https://arxiv.org/pdf/1804.03599.pdf).

    L = E[log p(x|z)] - g * |KL(q(z|x) || p(z)) - C|
      = recon_loss - gamma * tf.math.abs(kl_loss - C)

    Parameters
    ----------
    model : VAE
        Module that subclasses a tf.keras.Model that can encode,
        reparameterize and decode the input.
    x : tf.Tensor
        Batch to make a forward pass with.
    params : dict (default=None)
        Parameters/keyword arguments for computing the loss.

    Returns
    -------
    losses : dict
        recon_loss : Reconstruction loss, E[log p(x|z)]
        kl_loss : KL divergence, KL(q(z|x) || p(z))
        annealed_kl_loss : KL divergence, g * |KL(q(z|x) || p(z)) - C|
        elbo_loss : Evidence lower bound, recon_loss + beta * kl_loss
        loss : elbo_loss
    """
    recon_loss_fn = params.get('recon_loss_fn', recon_losses.bernoulli_loss)
    step = tf.summary.experimental.get_step()

    gamma = params.get('gamma', 1000)
    capacity = params.get('capacity', 25)
    iter_threshold = params.get('iter_treshold', 1000)

    C = np.minimum(capacity, capacity * step / iter_threshold)

    z_mean, z_logvar = model.encode(x)
    z_samp = model.reparameterize(z_mean, z_logvar)
    x_decoded = model.decode(z_samp)

    recon_loss = recon_loss_fn(x, x_decoded, from_logits=True)
    kl_loss = ops.compute_gaussian_kl(z_mean, z_logvar)
    annealed_kl_loss = gamma * tf.math.abs(kl_loss - C)
    elbo_loss = recon_loss + annealed_kl_loss

    losses = {
        'loss': elbo_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'annealed_kl_loss': annealed_kl_loss,
        'elbo_loss': elbo_loss
    }
    return losses


def compute_dip_vae_loss(model, x, params={}):
    """Equation  of "Variational Inference of Disentangled Latent Concepts
    from Unlabeled Observations" (https://arxiv.org/pdf/1711.00848.pdf).

    DIPVAE-I
    --------
    L = E[log p(x|z)] - KL(q(z|x) || p(z)) - l_od * sum(cov(mu(x)))^2
        - l_d * sum(cov(mu(x)) - 1)^2

    DIPVAE-II
    ---------
    L = E[log p(x|z)] - KL(q(z|x) || p(z)) - l_od * sum(cov(z))^2
        - l_d * sum(cov(z) - 1)^2

        = recon_loss - kl_loss - lambda_od * cov_z_off_diag
          - lambda_o * cov_z_on_diag

    Parameters
    ----------
    model : VAE
        Module that subclasses a tf.keras.Model that can encode,
        reparameterize and decode the input.
    x : tf.Tensor
        Batch to make a forward pass with.
    params : dict (default=None)
        Parameters/keyword arguments for computing the loss.

    Returns
    -------
    losses : dict
        recon_loss : Reconstruction loss, E[log p(x|z)]
        kl_loss : KL divergence, KL(q(z|x) || p(z))
        elbo_loss : Evidence lower bound, recon_loss + kl_loss
        regularizer_od : Off diagonal of covariance, D[Cov[z]]
        regularizer_d : Diagonal of covariance, OD[Cov[z]]
        dip_loss : DIPVAE loss, elbo_loss + regularizer_od + regularizer_d
        loss : dip_loss
    """
    regularizer_type = 'ii' if params.get('loss_fn') == 'dip_vae_ii' else 'i'
    lambda_d = params.get('lambda_d', 10)
    lambda_od = params.get('lambda_od', 10)
    recon_loss_fn = params.get('recon_loss_fn', recon_losses.bernoulli_loss)

    z_mean, z_logvar = model.encode(x)
    z_samp = model.reparameterize(z_mean, z_logvar)
    x_decoded = model.decode(z_samp)

    recon_loss = recon_loss_fn(x, x_decoded, from_logits=True)
    kl_loss = ops.compute_gaussian_kl(z_mean, z_logvar)
    elbo_loss = recon_loss + kl_loss

    cov_z_mean = ops.compute_cov_matrix(z_mean)
    if regularizer_type == 'i':
        cov_z_on_diag, cov_z_off_diag = ops.compute_on_off_diag(cov_z_mean)
    else:
        sigma = tf.linalg.diag(tf.exp(z_log_var))
        exp_cov = tf.reduce_mean(sigma, axis=0)
        cov_z = cov_z_mean + exp_cov
        cov_z_on_diag, cov_z_off_diag = ops.compute_on_off_diag(cov_z)

    regularizer_od = lambda_od * tf.reduce_sum(cov_z_off_diag**2)
    regularizer_d = lambda_d * tf.reduce_sum((cov_z_on_diag - 1)**2)
    dip_regularizer = regularizer_od + regularizer_d
    dip_loss = elbo_loss + dip_regularizer

    losses = {
        'loss': dip_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'elbo_loss': elbo_loss,
        'regularizer_od': regularizer_od,
        'regularizer_d': regularizer_d,
        'dip_regularizer': dip_regularizer,
        'dip_loss': dip_loss
    }
    return losses


def compute_factor_vae_loss(model, x, params={}):
    """Equation 2 of "Disentangling by Factorizing"
    (https://arxiv.org/pdf/1802.05983).

    L = E[log p(x|z)] - KL(q(z|x) || p(z)) - g * KL (q(z) || q~(z))
      = recon_loss - kl_loss - gamma * tc_loss

    s.t. q(z) is a true and q~(z) is a shuffled distribution of the sampled
    latent space.

    Parameters
    ----------
    model : FactorVAE
        Module that subclasses a tf.keras.Model that can encode,
        reparameterize, decode and discriminate the input.
    x : tf.Tensor
        Batch to make a forward pass with.
    params : dict (default=None)
        Parameters/keyword arguments for computing the loss.

    Returns
    -------
    losses : dict
        recon_loss : Reconstruction loss, E[log p(x|z)]
        kl_loss : KL divergence, KL(q(z|x) || p(z))
        elbo_loss : Evidence lower bound, recon_loss + kl_loss
        tc_loss : Total correlation, KL (q(z) || q~(z))
        factor_vae_loss : FactorVAE loss, elbo_loss + gamma * tc_loss
        disc_loss : Discriminator loss, E[log (q(z) / q~(z))]
        loss : factor_vae_loss
    """
    beta = params.get('beta', 1)
    gamma = params.get('gamma', 10)
    recon_loss_fn = params.get('recon_loss_fn', recon_losses.bernoulli_loss)

    z_mean, z_logvar = model.encode(x)
    z_samp = model.reparameterize(z_mean, z_logvar)
    x_decoded = model.decode(z_samp)
    z_perm = permute_dims(z_samp)
    z_logits, z_probs = model.discriminator(z_samp)
    z_perm_logits, z_perm_probs = model.discriminator(z_perm)

    recon_loss = recon_loss_fn(x, x_decoded, from_logits=True)
    kl_loss = ops.compute_gaussian_kl(z_mean, z_logvar)
    elbo_loss = recon_loss + beta * kl_loss

    tc_loss = tf.reduce_mean(z_logits[:,0] - z_logits[:,1])
    factor_vae_loss = elbo_loss + gamma * tc_loss
    # Algorithm 2 FactorVAE
    disc_loss = - 0.5 * tf.add(
        tf.reduce_mean(tf.math.log(z_probs[:, 0])),
        tf.reduce_mean(tf.math.log(z_perm_probs[:, 1]))
    )

    losses = {
        'loss': factor_vae_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'elbo_loss': elbo_loss,
        'tc_loss': tc_loss,
        'factor_vae_loss': factor_vae_loss,
        'disc_loss': disc_loss
    }
    return losses


def compute_tc_vae_loss(model, x, params={}):
    """Equation 4 of "Isolating Sources of Disentanglement in VAEs"
    (https://arxiv.org/pdf/1802.04942.pdf).

    L = E[log p(x|z)] - a * KL ( q(z|x) || q(x) ) - B * KL ( q(z) || prod q(z) )
        - g * sum ( KL ( q(z) || p (z) ) )

      = recon_loss - alpha * mi_loss - beta * tc_loss - gamma * kl_loss

    Parameters
    ----------
    model : VAE
        Module that subclasses a tf.keras.Model that can encode,
        reparameterize and decode the input.
    x : tf.Tensor
        Batch to make a forward pass with.
    params : dict (default=None)
        Parameters/keyword arguments for computing the loss.

    Returns
    -------
    losses : dict
        recon_loss : Reconstruction loss, E[log p(x|z)]
        mi_loss : Mutual Information, KL ( q(z|x) || q(x) )
        tc_loss : Total correlation, KL ( q(z) || prod q(z) )
        kl_loss : KL divergence, sum ( KL ( q(z) || p (z) ) )
        tcvae_loss : TCVAE loss, recon_loss + alpha * mi_loss
                                            + beta * tc_loss
                                            + gamma * kl_loss
        loss : tcvae_loss
    """
    beta = params.get('beta', 5)
    alpha = params.get('alpha', 1)
    gamma = params.get('gamma', 1)
    recon_loss_fn = params.get('recon_loss_fn', recon_losses.bernoulli_loss)

    z_mean, z_log_var = model.encode(x)
    z_samp = model.reparameterize(z_mean, z_log_var)
    x_decoded = model.decode(z_samp)

    recon_loss = recon_loss_fn(x, x_decoded, from_logits=True)

    log_qz = ops.compute_log_qz(z_samp, z_mean, z_log_var)
    log_prod_qz_i = ops.compute_log_prod_qz_i(z_samp, z_mean, z_log_var)
    log_qz_cond_x = ops.compute_log_qz_cond_x(z_samp, z_mean, z_log_var)
    log_pz = ops.compute_log_pz(z_samp, z_mean, z_log_var)

    mi_loss = tf.reduce_mean(log_qz_cond_x - log_qz)
    tc_loss = tf.reduce_mean(log_qz - log_prod_qz_i)
    kl_loss = tf.reduce_mean(log_prod_qz_i - log_pz)
    # equation 2 (ELBO TC-Decomposition)
    decomp_loss = (alpha * mi_loss
                   + beta * tc_loss
                   + gamma * kl_loss)

    tcvae_loss = recon_loss + decomp_loss

    losses = {
        'loss': tcvae_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'mi_loss': mi_loss,
        'tc_loss': tc_loss,
        'decomp_loss': decomp_loss,
        'tcvae_loss': tcvae_loss
    }

    return losses

