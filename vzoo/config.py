import vzoo.losses.disentangled_losses as losses
import vzoo.eval.disentangled_metrics as metrics
import vzoo.vis.utils as vis_utils
from vzoo.data import loaders


DATASET_TO_LATENT_INDICES = {
    'dsprites': [1, 2, 3, 4, 5],
    'cars': [0, 1, 2],
    'dummy': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

DATASET_TO_FACTOR_SIZES = {
    'dsprites': [1, 3, 6, 40, 32, 32],
    'cars': [4, 24, 183],
    'dummy': [1] * 10
}

DATASET_TO_SHAPE = {
    'dsprites': (64, 64, 1),
    'cars': (64, 64, 3),
    'celeba': (64, 64, 3),
    'dummy': 10
}

DATASET_TO_LOAD_FN = {
    'cars': loaders.load_cars,
	'celeba': loaders.load_celeba,
    'dsprites': loaders.load_dsprites,
    'dummy': loaders.load_test
}

LOSS_FN_DICT = {
    'elbo': losses.compute_elbo_loss,
    'beta_vae': losses.compute_beta_vae_loss,
    'factor_vae': losses.compute_factor_vae_loss,
    'beta_tcvae': losses.compute_tc_vae_loss,
    'dip_vae_i': losses.compute_dip_vae_loss,
    'dip_vae_ii': losses.compute_dip_vae_loss
}

SCORE_FN_DICT = {
    'bvae': metrics.compute_bvae_score,
    'sap': metrics.compute_sap_score,
    'mig': metrics.compute_mig_score,
    'dci': metrics.compute_dci_score,
    'mod': metrics.compute_mod_explicit_score
}

latent_traversal_fns = [
    vis_utils.gaussian_traversal,
    vis_utils.interval_traversal,
    vis_utils.linear_traversal
]

DISENTANGLED_LOSSES= [
    'recon_loss',
    'kl_loss',
    'mi_loss',
    'elbo_loss',
    'tc_loss',
    'factor_vae_loss',
    'disc_loss',
    'dip_regularizer',
    'dip_loss',
    'loss',
]

DISENTANGLED_SCORES = [
    'bvae_score',
    'sap_score',
    'mig_score',
    'mod_score',
    'explicit_score',
    'dci_comp_score',
    'dci_info_score',
    'dci_dis_score'
]
