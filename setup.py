from setuptools import setup
from setuptools import find_packages

config = {
    'name': 'vzoo',
    'version': '1.0',
    'license': 'MIT',
    'description': 'Variational inference and disentangled representation through unsupervised learning.',
    'author': 'a ghost',
    'author_email': 'tannercarbonati@gmail.com',
    'packages': find_packages(),
    'install_requires': [
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scikit-image',
        'scipy',
        'pillow',
        'tensorflow-gpu',
        'tqdm',
        'umap-learn',
        'tf-nightly',
    ]
}

setup(**config)
