# Super Convergence

> WIP: I'll post notebooks in a sec ...

Playing with ideas from:

- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) [1]
- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820) [2]
- [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120) [3]


## Notes

### Cycle length

Number of `iterations = dataset_size / batch_size`. Length of cycle is robust but setting it to 2-10 times the iterations in a epoch.

```
dataset_size = 50_000
batch_size = 100
iters = dataset_size / batch_size # 500
stepsize = 2 * iters # 1000
```

This means the learning rate will increase for 1000 batches of data and then decrease the next 1000.

### Finding learning rate

A larger learning rate helps avoid overfitting but it can also lead to divergence.

The maximum bound is found through a pre-training run where the test/validation is measured while the
learning rate is increased. The learning rate at which the test/validation loss begins to increase is
the chosen as the maximum learning rate.

There are several ways one can choose the minimum learning rate bound: 

1. a factor of 3 or 4 less than the maximum bound 
2. a factor of 10 or 20 less than the maximum bound if only one cycle is used 
3. by a short test of hundreds of iterations with a few initial learning rates and pick the largest one that allows
convergence to begin without signs of overfitting as shown in Figure 1a (if the initial learning rate
is too large, the training won’t begin to converge).

### Batch Size / Weight Decay

The largest batch size should be used. Batch size and weight dcay should be tested while the learning rate is found via a grid search. Complex architectures require less regularization (1e-4, 1e-5, 1e-6), while simpler architectures could use more (1e-2, 1e-3, 1e-4).

### Momentum

A constant momentum works quite well. If a cycle is used it should be "upside-down", i.e. go from max momentum to
min momentum and back to max momentum. Searching for an optimal momentum isn't as easy as a learning rate.


### 1cycle

Always use one cycle that is smaller than the total number of iterations/epochs and allow the learning rate to
decrease several orders of magnitude less than the initial learning rate for the remaining iterations/epochs.

Say we have 100 epochs, a dataset size of 50,000 and a batch size of 500 - 100 iterations in an epoch. We have 10,000 iterations to do with as we please. Using 1cycle we could use 90,000 iterations for a cycle. This means the step size would be 45,000. The remaining 10,000 iterations would anneal from the minimum learning rate to 0.

In [3] 85-90% of the iterations are used in the cycle.
