# PyTorch vs Keras

Compare [PyTorch](http://pytorch.org/) and [Keras](https://keras.io/)(TensorFlow backend).

Train MNIST for one epoch.

Python 3.6.1

## Differences

### Loss function
`mnist_keras.py` uses `categorical_crossentropy` as loss function.

On the other hand, PyTorch does not have equivalent function. So I calculate it by myself.

```python
loss = -out.gather(1, label).log().mean()
```

By the way, in PyTorch, it is common to use `CrossEntropy` which uses `LogSoftmax`.  
[Note of Softmax](http://pytorch.org/docs/master/nn.html#softmax) says
> This module doesn’t work directly with NLLLoss, which expects the Log to be computed between the Softmax and itself. Use Logsoftmax instead (it’s faster).

This repository contains both versions of them(`mnist_pytorch.py` and `mnist_pytorch_logsoftmax.py`).
