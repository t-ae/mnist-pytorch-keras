#!/usr/bin/env python

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn
import torch.utils.data
from torch.autograd import Variable
from keras.datasets import mnist


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.linear1 = nn.Linear(64*7*7, 512)
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, 0.5)
        x = self.linear2(x)

        return x


def main(use_cuda, verbose):
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255
    x_train = x_train[..., None]
    x_train = x_train.transpose([0, 3, 1, 2])
    x_train = x_train.astype(np.float32)
    x_train = np.ascontiguousarray(x_train)
    y_train = y_train.astype(np.int64)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    batch_size = 32
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)

    model = Net()
    opt = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        model.cuda()
        criterion.cuda()

    # measure
    start = time.time()
    for i, data in enumerate(loader):
        opt.zero_grad()

        batch, label = data
        batch = Variable(batch)
        label = Variable(label)
        if use_cuda:
            batch.cuda()
            label.cuda()

        out = model.forward(batch)

        loss = criterion(out, label)
        loss.backward()
        opt.step()

        if verbose:
            print(f"{i}: {loss.data[0]}", end=" "*16+"\r")

    print(f"Elapsed time: {time.time()-start}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use CUDA')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
    args = parser.parse_args()
    main(args.use_cuda, args.verbose)
