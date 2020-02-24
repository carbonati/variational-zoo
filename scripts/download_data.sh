#!/bin/bash
mkdir -p data

echo "Downloading MNIST data."
if [[ ! -d "data/mnist" ]]; then
    mkdir -p data/mnist

    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O data/mnist/train-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O data/mnist/t10k-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O data/mnist/train-labels-idx1-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O data/mnist/t10k-labels-idx1-ubyte.gz
    gunzip data/mnist/train-images-idx3-ubyte.gz
    gunzip data/mnist/t10k-images-idx3-ubyte.gz
    gunzip data/mnist/train-labels-idx1-ubyte.gz
    gunzip data/mnist/t10k-labels-idx1-ubyte.gz
fi
echo "Finished downloading MNIST."

# echo "Downloading Cifar-10 data."
# if [[ ! -d "data/cifar" ]]; then
#     mkdir -p data/cifar
#     wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#     tar -zxvf cifar-10-python.tar.gz
#     mv cifar-10-batches-py data/cifar
#     rm -f cifar-10-python.tar.gz
# fi
# echo "Finished downloading Cifar-10"

echo "Downloading CelebA data."
if [[ ! -d "data/celeba" ]]; then
    mkdir -p data/celeba
    wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip .
    unzip celeba.zip -d data/celeba
    rm celeba.zip
fi
echo "Finished downloading CelebA."

echo "Downloading dSprites data."
if [[ ! -d "data/dsprites" ]]; then
  mkdir -p data/dsprites
  wget https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz -O data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
fi
echo "Finished Downlaoding dSprites."


echo "Downloading Cars3D data."
if [[ ! -d "data/cars" ]]; then
    wget http://www.scottreed.info/files/nips2015-analogy-data.tar.gz -O nips2015-analogy-data.tar.gz
    tar xzf nips2015-analogy-data.tar.gz
    rm nips2015-analogy-data.tar.gz
fi
echo "Finished downloading Cars3D."
