import numpy as np
from scipy.stats import norm


def linear_traversal(z,
                     num_traversals,
                     z_min=-2.5,
                     z_max=2.5,
                     scalar=2):
    """Applies a linear traversal to a latent variable."""
    z_traversed = np.linspace(z_min + z,
                              z_max + z,
                              num=num_traversals,
                              endpoint=False)
    return np.squeeze(z_traversed)


def interval_traversal(z,
                       num_traversals,
                       z_min=-2.5,
                       z_max=2.5,
                       scalar=2):
    """Applies a interval traversal to a latent variable."""
    z_start = (z - z_min) / (z_max - z_min)
    z_traversed = np.linspace(z_start,
                              z_start + scalar,
                              num=num_traversals,
                              endpoint=False)
    z_traversed -= np.maximum(0, scalar * z_traversed - scalar)
    z_traversed += np.maximum(0, -scalar * z_traversed)
    z_traversed *= (z_max - z_min) + z_min
    return np.squeeze(z_traversed)


def gaussian_traversal(z, num_traversals, loc=0., scale=1.25, scalar=2):
    """Appliees a gaussian fit traversal to a latent variable."""
    cdf = norm.cdf(z, loc=loc, scale=scale)
    z_traversed = np.linspace(cdf,
                              cdf + scalar,
                              num=num_traversals,
                              endpoint=False)
    z_traversed -= np.maximum(0, scalar * z_traversed - scalar)
    z_traversed += np.maximum(0, -scalar * z_traversed)
    z_traversed = np.clip(z_traversed, 0, 1)
    z_traversed = [norm.ppf(q, loc=loc, scale=scale) for q in z_traversed]
    return np.squeeze(z_traversed)


def pad_images(images, pad_size=1, pad_value=0, axis=0):
    """Pads and concatenates a list of images."""
    num_images = len(images)
    pad_shape = list(images[0].shape)
    pad_shape[axis] = pad_size
    x_pad = np.ones(pad_shape, dtype=images[0].dtype) * pad_value

    images_padded = []
    for i, img in enumerate(images):
        images_padded.append(img)
        if i < num_images - 1:
            images_padded.append(x_pad)

    return np.concatenate(images_padded, axis=axis)


def pad_grid(images, num_samples, pad_size, pad_value):
    """
    Generates a grid of padded images where each columns is a sample and each
    row is a latent.
    """
    num_images = len(images)
    cols = int(np.ceil(num_images / num_samples))
    rows = int(np.ceil(num_images / cols))
    grid = []

    for i in range(rows):
        grid.append(pad_images(images[i*cols : (i+1)*cols],
                               pad_size,
                               pad_value=pad_value,
                               axis=1))
    return pad_images(grid, pad_size, pad_value=pad_value, axis=2)
