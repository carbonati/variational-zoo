import numpy as np
import imageio
from skimage import img_as_ubyte
import tensorflow as tf
from vzoo.vis.utils import pad_grid


def generate_traversal_grid(model,
                            dataset,
                            traversal_fn,
                            latent_indices,
                            num_samples,
                            num_traversals,
                            random_state=None):
    """Generates a latent traversed grid of images."""
    random_state = random_state or np.random.RandomState()

    grid = []
    for i in range(num_samples):
        img_idx = random_state.randint(len(dataset))
        image = np.expand_dims(dataset[img_idx].astype(np.float32),
                               axis=0)
        representation = model.encode(image)[0].numpy()

        for latent_idx in latent_indices:
            representations = np.repeat(representation,
                                        num_traversals,
                                        axis=0)
            representations[:, latent_idx] = traversal_fn(
                representation[:,latent_idx],
                num_traversals=num_traversals
            )
            x_decoded = tf.nn.sigmoid(model.decode(representations)).numpy()
            grid.append(x_decoded)
    return grid

def save_latent_traversal(filepath,
                          model,
                          dataset,
                          traversal_fn,
                          latent_indices,
                          num_traversals=24,
                          num_samples=8,
                          random_state=None,
                          params=None):
    """Generates, pads, and saves a latent traversed grid of images."""
    params = params or {}
    pad_size = params.get('pad_size', 1)
    pad_value = params.get('pad_value', 0)
    fps = int(num_traversals // 2)

    grid = generate_traversal_grid(model,
                                   dataset,
                                   traversal_fn,
                                   latent_indices,
                                   num_samples,
                                   num_traversals,
                                   random_state=random_state)
    grid = pad_grid(grid, num_samples, pad_size, pad_value)

    imageio.mimsave(filepath, img_as_ubyte(grid), fps=fps)
