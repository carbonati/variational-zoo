import numpy as np
from functools import partial
import tensorflow as tf
from sklearn.model_selection import train_test_split
from vzoo.data.sampling import LatentGenerator
from vzoo import config
from vzoo.config import (DATASET_TO_LOAD_FN,
                           DATASET_TO_SHAPE,
                           DATASET_TO_FACTOR_SIZES,
                           DATASET_TO_LATENT_INDICES)


class BaseDataset(object):
    """Base disentangled dataset class."""

    def __init__(self, root):
        self.root = root
        self.load_fn = None
        self.input_shape = None
        self.factor_sizes = None
        self.latent_indices = None
        self.images = None
        self.generator = None
        self.num_samples = None
        self.batch_size = None
        self.seed = None

    @property
    def num_latents(self):
        """Returns the number of latent variables."""
        raise NotImplementedError

    @property
    def num_factors(self):
        """Returns the number of ground truth factors."""
        raise NotImplementedError

    @property
    def shape(self):
        """Returns the shape of the dataset."""
        raise NotImplementedError

    def __len__(self):
        return self.shape[0]

    def load_data(self):
        """Logic to load data lives here."""
        raise NotImplementedError

    def sample(self, num_samples, random_state=None):
        """Generates a set of ground truth latent factors & the observations
        they generate.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        random_state : np.random.RandomState
            Pseudo-random number generator.

        Returns
        -------
        factors : np.ndarray
            Ground truth factors of shape (num_samples, self.num_factors)
            generated from `observations`.
        observations : np.ndarray
            observations of shape (num_samples, self.input_shape) used to
            generate `factors`.
        """
        raise NotImplementedError

    def random_split(self, lengths=None, test_size=None, random_state=None):
        """
        Randomly split a dataset into non-overlapping new datasets of given size.

        Parameters
        ----------
        lengths : list
            List of number of samples in non-overlapping dataset
        test_size : float (default=None)
            If no `lengths` are passed in splits the data into a train/test set
            of sizes (1-test_size)/test_size.
        random_state : np.random.RandomState
            Pseudo-random number generator.

        Returns
        -------
        Non overlapping datasets with the number of samples passed into
        `lengths` or specified by `test_size`.
        """
        random_state = random_state or np.random.RandomState(self.seed)
        if lengths is None:
            test_size = test_size if test_size is not None else 0.2
            test_length = int(np.ceil(len(self) * test_size))
            lengths = [len(self) - test_length, test_length]

        ind = random_state.choice(len(self),
                                  size=np.sum(lengths),
                                  replace=False)
        return [self.prepare_data(ind[offset - length : offset])
                for offset, length
                in zip(np.cumsum(lengths), lengths)]

    def prepare_data(self, indices=None):
        """Returns input matrix as a tf.data.Dataset.

        Parameters
        ----------
        indices : list, np.ndarray
            Indicies of images to convert to a tf.data.Dataset. If
            no indices left as None the full dataset will be used.

        Returns
        -------
        dataset : tf.data.Dataset
            A subset of or all images loaded.
        """
        indices = range(len(self)) if indices is None else indices
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(self.images[indices], dtype=tf.float32)
        )
        dataset = dataset.shuffle(len(indices)).batch(self.batch_size)
        return dataset


class DisentangledDataset(BaseDataset):
    """Disentangled dataset.

    Used to load images from a dataset, split train/test sets, and sample
    latent factors to compute various disentanglement scores.

    Parameters
    ----------
    root : str
        Path to root directory of a dataset.
    load_fn : function
        Function to load a dataset from `root`.
    input_shape : tuple/list
        Shape of images.
    factor_sizes : list (default=None)
        Number of distinct values a ground truth latent factor can take on.
    latent_indices : None (default=None)
        Indices of latent factors mapped to `factor_sizes`.
    batch_size : int
        Number of samples per batch.
    num_samples :
        Number of samples to load. If left as None all images will be used.
    seed : int
        Random state.
    """
    def __init__(self,
                 root,
                 load_fn,
                 input_shape,
                 factor_sizes=None,
                 latent_indices=None,
                 batch_size=64,
                 num_samples=None,
                 seed=420):
        super(DisentangledDataset, self).__init__(root)
        self.load_fn = load_fn
        self.input_shape = input_shape
        self.factor_sizes = factor_sizes
        self.latent_indices = latent_indices
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.seed = seed

        if (latent_indices is not None) and (factor_sizes is not None):
            self.generator = LatentGenerator(self.latent_indices,
                                             self.factor_sizes,
                                             seed=self.seed)
        self.images = self.load_data()

    @property
    def num_latents(self):
        return len(self.latent_indices)

    @property
    def num_factors(self):
        return len(self.factor_sizes)

    @property
    def shape(self):
        return self.images.shape

    def __getitem__(self, idx):
        return self.images[idx].reshape(self.input_shape)

    def load_data(self):
        return self.load_fn(root=self.root,
                            input_shape=self.input_shape,
                            factor_sizes=self.factor_sizes,
                            generator=self.generator,
                            num_samples=self.num_samples)

    def sample_factors_of_variation(self, num_samples, random_state=None):
        """Randomly samples a batch of factors."""
        return self.generator.sample_latent_factors(num_samples,
                                                    random_state)

    def sample_observations_from_factors(self, factors, random_state=None):
        """Randomly samples a batch of observations from a batch of factors."""
        all_factors = self.generator.sample_all_factors(factors,
                                                        random_state)
        indices = self.generator.feature_lookup[all_factors]
        return self.images[indices].astype(np.float32)

    def sample(self, num_samples, random_state=None):
        factors = self.sample_factors_of_variation(num_samples,
                                                   random_state=random_state)
        observations = self.sample_observations_from_factors(
            factors,
            random_state=random_state
        )
        return factors, observations

    def sample_observations(self, num_samples, random_state=None):
        return self.sample(num_samples, random_state)[1]


class dSprites(DisentangledDataset):
    """dSprites: Disentanglement testing Sprites dataset.
    (https://github.com/deepmind/dsprites-dataset/)

    737280 images of shape (64, 64, 1).

    Parameters
    ----------
    root : str
        Root directory to images.
    batch_size : int (default=64)
        Number of samples per batch.
    num_samples :
        Number of samples to load. If left as None all images will be used.
    """
    def __init__(self,
                 root='data/dsprites',
                 batch_size=64,
                 num_samples=None):
        self.dataset = 'dsprites'
        super(dSprites, self).__init__(
            root=root,
            load_fn=DATASET_TO_LOAD_FN[self.dataset],
            input_shape=DATASET_TO_SHAPE[self.dataset],
            factor_sizes=DATASET_TO_FACTOR_SIZES.get(self.dataset),
            latent_indices=DATASET_TO_LATENT_INDICES.get(self.dataset),
            batch_size=batch_size,
            num_samples=num_samples
        )


class Cars3D(DisentangledDataset):
    """Cars3D.

    "Weakly-supervised Disentangling with Recurrent Transformations for 3D View
    Synthesis" (https://arxiv.org/pdf/1601.00706.pdf)

    Parameters
    ----------
    root : str
        Root directory to images.
    batch_size : int (default=64)
        Number of samples per batch.
    num_samples : int (default=None)
        Number of samples to load. If left as None all images will be used.
    """
    def __init__(self,
                 root='data/cars',
                 batch_size=64,
                 num_samples=None):
        self.dataset = 'cars'
        super(Cars3D, self).__init__(
            root=root,
            load_fn=DATASET_TO_LOAD_FN[self.dataset],
            input_shape=DATASET_TO_SHAPE[self.dataset],
            factor_sizes=DATASET_TO_FACTOR_SIZES.get(self.dataset),
            latent_indices=DATASET_TO_LATENT_INDICES.get(self.dataset),
            batch_size=batch_size,
            num_samples=num_samples
        )


class CelebaDataset(DisentangledDataset):
    """CelebFaces Attributes Dataset (CelebA).
    (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

    202599 images of shape (64, 64, 3).

    Parameters
    ----------
    root : str
        Root directory to images.
    batch_size : int (default=64)
        Number of samples per batch.
    num_samples :
        Number of samples to load. If left as None all images will be used.
    """
    def __init__(self,
                 root='data/celeba/celeba_processed',
                 batch_size=64,
                 num_samples=None):
        self.dataset = 'celeba'
        super(CelebaDataset, self).__init__(
            root=root,
            load_fn=config.DATASET_TO_LOAD_FN[self.dataset],
            input_shape=config.DATASET_TO_SHAPE[self.dataset],
            factor_sizes=config.DATASET_TO_FACTOR_SIZES.get(self.dataset),
            latent_indices=config.DATASET_TO_LATENT_INDICES.get(self.dataset),
            batch_size=batch_size,
            num_samples=num_samples
        )

    def __len__(self):
        return len(self.images.index_array)

    def _generator(self, indices):
        """Utility used to subset a generator with `indices`."""
        for i in indices:
            yield self.images[i]

    def prepare_data(self, indices=None):
        """Returns input matrix as a tf.data.Dataset.

        Parameters
        ----------
        indices : list, np.ndarray
            Indicies of images to convert to a tf.data.Dataset. If
            no indices left as None the full dataset will be used.

        Returns
        -------
        dataset : tf.data.Dataset
            A subset of or all images loaded.
        """
        indices = range(len(self)) if indices is None else indices

        f = partial(self._generator, indices=indices)
        dataset = tf.data.Dataset.from_generator(f,
                                                 tf.float32,
                                                 output_shapes=(1,64,64,3))
        return dataset.unbatch().batch(self.batch_size)


class TestDataset(BaseDataset):
    """Dataset for unit testing."""

    def __init__(self,
                 root=None,
                 batch_size=16,
                 seed=420):
        super(TestDataset, self).__init__(root)
        self.load_fn = load_fn
        self.input_shape = input_shape
        self.factor_sizes = factor_sizes
        self.latent_indices = latent_indices
        self.batch_size = batch_size
        self.seed = seed

        self.images = self.load_data()

    @property
    def num_latents(self):
        return len(self.latent_indices)

    @property
    def num_factors(self):
        return len(self.factor_sizes)

    @property
    def shape(self):
        return self.images.shape

    def load_data(self):
        return self.load_fn(input_shape=(10,1))

    def sample_factors_of_variation(self, num_samples, random_state):
        """Randomly samples a batch of factors."""
        return random_state.randint(self.num_factors,
                                    size=(num_samples, self.num_factors))

    def sample_observations_from_factors(self, factors, random_state):
        """Randomly samples a batch of observations from a batch of factors."""
        return factors

    def sample(self, num_samples, random_state=None):
        factors = self.sample_factors_of_variation(num_samples,
                                                   random_state=random_state)
        observations = self.sample_observations_from_factors(
            factors,
            random_state=random_state
        )
        return factors, observations
