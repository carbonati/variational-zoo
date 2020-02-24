import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from skimage.io import imread
from skimage.transform import resize
import multiprocessing as mp
from multiprocessing import Pool
import _pickle as cPickle
import tensorflow as tf


def load_dsprites(root,
                  filename='dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                  num_samples=None,
                  **kwargs):
    """dSprites: Disentanglement testing Sprites dataset
    (https://github.com/deepmind/dsprites-dataset/)

    Parameters
    ----------
    root : str
        Root directory to images.
    filename : 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        Name of file in `root` to load.

    Returns
    -------
    images : np.ndarray
        Images of shape (737280, 64, 64, 1).
    """
    if not root.endswith(filename):
        root = os.path.join(root, filename)
    with tf.io.gfile.GFile(root, 'rb') as f:
        data = np.load(f, allow_pickle=True, encoding='latin1')
    images = np.array(data['imgs'])

    if num_samples is not None:
        ind = list(range(len(images)))
        filepaths = np.random.choice(ind,
                                     size=num_samples,
                                     replace=False)
        images = iamges[ind]

    return np.expand_dims(images, axis=-1)


def _load_cars(filename, output_shape=(64, 64, 3)):
    with tf.io.gfile.GFile(filename, "rb") as f:
        mat = loadmat(f)
    data = np.einsum("abcde->deabc", mat['im'])
    data = data.reshape((-1,) + data.shape[2:])
    imgs = np.zeros((data.shape[0],) + output_shape)
    for i in range(len(imgs)):
        imgs[i] = resize(data[i], output_shape)
    return imgs


def load_cars(root,
              output_shape,
              factor_sizes,
              generator,
              num_samples=None,
              **kwargs):
    """Cars3D

    "Weakly-supervised Disentangling with Recurrent Transformations for 3D View
    Synthesis" (https://arxiv.org/pdf/1601.00706.pdf)
    """
    factor_sizes = factor_sizes or [4, 24, 183]
    filepaths = glob.glob(os.path.join(root, '*.mat'))
    if num_samples is not None:
        filepaths = np.random.choice(filepaths,
                                     size=num_samples,
                                     replace=False)
    factor_1 = range(factor_sizes[0])
    factor_2 = range(factor_sizes[1])
    images = np.zeros(shape=(np.prod(factor_sizes),) + output_shape)

    for i, filepath in enumerate(tqdm(filepaths, desc='Loading cars dataset')):
        sub_images = _load_cars(filepath)
        all_factors = np.transpose([
            np.tile(factor_1, factor_sizes[1]),
            np.repeat(factor_2, factor_sizes[0]),
            np.tile(i, np.prod(factor_sizes[:2]))
        ])
        indices = generator.factor_lookup[all_factors]
        images[indices] = sub_images

    return images


def load_celeba(root,
                rescale=1./255,
                input_shape=(64, 64, 3),
                num_samples=None,
                **kwargs):
    """CelebFaces Attributes Dataset (CelebA)
    (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

    Parameters
    ----------
    root : str
        Root directory to images.
    rescale: float (default=1/255)
        Factor used to scale image.
    input_shape : tuple/list (default=(64, 64, 3))
        Expected shape of each image to resize if necessary.
    num_samples : int (default=None)
        Number of samples to load.
            If left as None all will be used.

    Returns
    -------
    generator : tf.keras.preprocessing.image.DirectoryIterator
        Image generator of size (202599, 64, 64, 3).
    """
    root_path, root_dir = os.path.split(root.strip('/'))
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=rescale,
                                                              dtype=tf.float32)
    generator = datagen.flow_from_directory(root_path,
                                            target_size=input_shape[:2],
                                            batch_size=1,
                                            classes=[root_dir],
                                            shuffle=False,
                                            class_mode=None)
    generator._set_index_array()
    return generator


def load_test(output_shape=(10,), num_samples=100):
    return np.zeros((num_samples,) + output_shape)

def _load_pickled(filepath):
    with open(filepath, 'rb') as f:
        data = cPickle.load(f, encoding='latin-1')
    return data

