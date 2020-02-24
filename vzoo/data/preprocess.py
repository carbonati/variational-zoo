import os
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from functools import partial


def _preprocess_celeba(filename, root, output_dir, shape, img_format):
    img = Image.open(os.path.join(root, filename))
    img = img.resize(shape, Image.ANTIALIAS)
    img.save(os.path.join(output_dir, filename), img_format)


def preprocess_celeba(root,
                      output_dir,
                      img_format='JPEG',
                      shape=(64, 64),
                      workers=None):
    """Preprocesses Celeba images to format used for modeling."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = os.listdir(root)
    print(f'Preprocessing {len(filenames)} images from {root} ',
          f'and saving to {output_dir}')

    f = partial(_preprocess_celeba,
                root=root,
                output_dir=output_dir,
                shape=shape,
                img_format=img_format)

    if workers is None:
        workers = int(mp.cpu_count() // 2) # sorry
    with mp.Pool(workers) as pool:
        list( tqdm(pool.imap_unordered(f, filenames),
                  desc='Preprocessing Celeba images',
                  total=len(filenames)) )
    print('Finished.')
