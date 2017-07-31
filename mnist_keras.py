#!/usr/bin/env python

import argparse
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, InputLayer
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical


def main(verbose):
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255
    x_train = x_train[..., None]
    y_train = to_categorical(y_train)

    model = Sequential([
        InputLayer([28, 28, 1]),
        Conv2D(32, 5, padding="same", activation="relu"),
        MaxPool2D(),
        Conv2D(32, 5, padding="same", activation="relu"),
        MaxPool2D(),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])

    model.compile("adam", "categorical_crossentropy")

    # measure
    start = time.time()
    model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=verbose, shuffle=False)
    print(f"Elapsed time: {time.time()-start}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use CUDA')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
    args = parser.parse_args()

    device = "/gpu:0" if args.use_cuda else "/cpu:0"
    with tf.device(device):
        main(args.verbose)
