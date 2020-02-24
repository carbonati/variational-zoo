import numpy as np
import random
from sklearn.utils.extmath import cartesian


class FeatureIndex(object):
    """Serves as a lookup dictionary that returns the index of an image given
    a factor configuration.

    Parameters
    ----------
    factor_sizes : list, np.ndarray
        List of integers where the i^th entry denotes the distinct number of
        values the i^th latent factor can take on.

    """
    def __init__(self, factor_sizes, features=None):
        self.factor_sizes = factor_sizes
        self.features = features
        self._num_feature_values = np.prod(self.factor_sizes)
        self.factor_bases = np.divide(self._num_feature_values,
                                      np.cumprod(self.factor_sizes))
        self._features_to_index = np.arange(self._num_feature_values)

    def _get_feature_space(self, features):
        return np.dot(features, self.factor_bases).astype(np.int32)

    def __len__(self):
        return len(self._features_to_index)

    def __getitem__(self, features):
        """
        Given a batch of ground truth latent factors returns the indices
        of the images they generate.
        """
        return self._features_to_index[self._get_feature_space(features)]

    def keys(self):
        return self._features_to_index

    def values(self):
        return self.features

    def items(self):
        return zip(self.keys(), self.values())

class LatentGenerator(object):
    """Latent factor of variation generator.

    Parameters
    ----------
    latent_indices : list, np.ndarray
        List of integers where each value denotes the index of a ground
        truth latent factor.
    factor_sizes : list, np.ndarray
        List of integers where the i^th entry denotes the distinct number of
        values the i^th latent factor can take on.
    seed : int
        Random state.
    """
    def __init__(self, latent_indices, factor_sizes, seed=420):
        self.latent_indices = latent_indices
        self.factor_sizes = factor_sizes
        self.num_factors = len(self.factor_sizes)
        self.num_latents = len(self.latent_indices)

        self.observed_factor_indices = self._get_observed_indices()
        self.num_observed_factors = len(self.observed_factor_indices)

        self.features = self._get_features()
        self.feature_lookup = FeatureIndex(self.factor_sizes, self.features)

        self.seed = seed
        self._random_state = np.random.RandomState(seed)


    def _get_observed_indices(self):
        indices = [
            i
            for i
            in range(self.num_factors)
            if i not in self.latent_indices
        ]
        return indices

    def _get_features(self):
        return cartesian([np.array(list(range(i)))
                          for i in
                          self.factor_sizes])

    def sample_latent_factors(self, num_samples, random_state=None):
        """Randomly samples a factor of variation for each latent index.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate a random latent space for.
        random_state : np.random.RandomState
            Pseudo-random number generator.

        Returns
        -------
        factors : np.ndarray
            A matrix of size (num_samples, self.num_latents) with randomly
            generated ground truth latent factors.
        """
        random_state = random_state or self._random_state
        factors = np.zeros((num_samples, self.num_latents))
        for pos, idx in enumerate(self.latent_indices):
            factors[:, pos] = self._sample_factors(idx,
                                                   num_samples,
                                                   random_state)
        return factors

    def sample_all_factors(self, latent_factors, random_state=None):
        """Randomly samples any additional latent factor

        Only concats additional latent factors if
            `self.num_latents` < `self.num_factors`.
        This typically happens when a factor of variation only has one distinct
        value (like dSprites), therefore, there is no need to use the index of
        that latent factor to randomly sample from when computing metrics
        like beta-VAE.

        Parameters
        ----------
        latent_factors : np.ndarray
            A matrix of size (num_samples, self.num_latents) with randomly
            generated ground truth latent factors. This is typically from the
            output of `self.sample_latent_factors`.
        random_state : np.random.RandomState
            Pseudo-random number generator.

        Returns
        -------
        all_factors : np.ndarray
            Matrix of size (num_samples, self.num_factors) with randomly
            generated ground truth latent factors.
        """
        random_state = random_state or self._random_state
        if self.num_observed_factors > 0:
            num_samples = len(latent_factors)
            all_factors = np.zeros((num_samples, self.num_factors))
            all_factors[:, self.latent_indices] = latent_factors
            for idx in self.observed_factor_indices:
                all_factors[:, idx] = self._sample_factors(idx,
                                                           num_samples,
                                                           random_state)
            return all_factors
        else:
            return latent_factors

    # change this to sample the range np.linspace()
    def _sample_factors(self, idx, size, random_state):
        return random_state.randint(self.factor_sizes[idx], size=size)

