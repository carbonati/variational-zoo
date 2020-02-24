# Variational Zoo (vzoo)


This project aims to promote reproducibility and ease of exploration for research and development of disentangled representations through unsupervised learning.


## Structure of vzoo
vzoo is disentangled into several modules

- `core`: Contains a `Trainer` used to optimize, evaluate, and visualize an unsupervised model. Commonly benchmarked networks for learning disentangled representations lives here as well.
- `data`: Data loaders for common datasets used to evaluate disentanglement, which includes
  - dSprites: [Disentanglement testing Sprites dataset](https://github.com/deepmind/dsprites-dataset)
  - Cars3D: [Weakly-supervised Disentangling with Recurrent Transformations for 3D View Synthesis](https://arxiv.org/pdf/1601.00706.pdf)
  - Celeba: [CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - MNIST: [The MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
  - SVHN: [The Street View House Numbers](http://ufldl.stanford.edu/housenumbers/)
A base disentangled dataset class to support new datasets for modeling lives here as well.
- `eval`: Evaluation metrics to quantify disentanglement logged to tensorboard. Common metrics include
  - beta-VAE: [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl)
  - factorVAE: [Disentangling by Factorizing](https://arxiv.org/pdf/1802.05983)
  - Disentanglement, Completeness, and Informativeness (DCI): [A Framework for the Quantitative Evaluation of Disentangled Representations](https://openreview.net/pdf?id=By-7dz-AZ)
  - Mutual Information Gap (MIG): [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/pdf/1802.04942.pdf)
  - Modularity and Explicitness: [Learning Deep Disentangled Embeddings with the F-Statistic Loss](https://arxiv.org/pdf/1802.05312.pdf)
  - Separated Attribute Predictability (SAP): [Variational Inference of Disentangled Latent Concepts from Unlabelled Observations](https://arxiv.org/pdf/1711.00848.pdf)
- `losses`: Losses, regularizations, and operations, which includes
  - elbo: [Auto-Encoding Variationa Bayes](https://arxiv.org/pdf/1312.6114.pdf)
  - beta-VAE: [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl)
  - factorVAE: [Disentangling by Factorizing](https://arxiv.org/pdf/1802.05983)
  - TC-VAE: [Isolating Sources of Disentanglement in VAEs](https://arxiv.org/pdf/1802.04942.pdf)
  - DIP-VAE: [Variational Inference of Disentangled Latent Concepts from Unlabeled Observations](https://arxiv.org/pdf/1711.00848.pdf)
  - InfoGan: [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/pdf/1606.03657.pdf)
  - Wasserstein: [Wasserstein Auto-Encoders](https://arxiv.org/pdf/1711.01558.pdf)
- `models`: Each unsupervised model takes in a function for each network required for training, where each function defines the networks architecture as shown in [network.py](https://www.youtube.com/watch?v=G_3sS39GkcQ). Each network is compiled using a function from [builders.py](https://github.com/carbonati/variational-zoo/blob/master/vzoo/models/builders.py) to allow easier use of using multiple optimizers.
- `vis`: Tools to visualize latent traversals and manifold representations.

<p align="center">
  <img width="500" height="500" src=assets/vaezoo_celeba.gif>
</p>

## Install
```
git clone git@github.com:carbonati/variational-zoo.git
cd variational-zoo
pip install .
```

### Collab example
tired, but will add soon

### Todo
- Add scripts to execute experiments from previous papers for reproducibility.
- Add unit testing for methods, metrics, and datasets.
- Create google collab notebook with example.

### Remarks

This project was inspired by a wonderful paper [Challenging Common Assumptions in the Unsupervised Learning of
Disentangled Representations](https://arxiv.org/pdf/1811.12359.pdf), along with many others mentioned in CITATIONS.md with hopes of expanding upon previous work and serving as a home for new research in disentangled representations through unsupervised learning and [representation learning](https://arxiv.org/pdf/1206.5538.pdf) as a whole.

Each scoring function attempts to follow the same notation as presented in each paper. In hindsight this may have been a poor decision as the notation across papers changes significantly, which is rather unfortunate, however, the consistency between the papers and the code may help with understanding one another better.

This project is a work in progress and continuously developed, please don't pet the representations.

