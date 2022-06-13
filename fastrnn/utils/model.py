from fastrnn.utils.text_processor import TextProcess
from fastrnn.utils.model_config import hidden_size, seq_len, learning_rate
import numpy as np
from numba import njit


@njit(fastmath=True, parallel=True)
def update_ht_jit(Wxh, x, Whh, h, bh, t, vocab_size):
    res = np.tanh(np.dot(Wxh, x[t].reshape(vocab_size, 1)) + np.dot(Whh, h[t - 1].reshape(hidden_size, 1)) + bh)
    return res


@njit(fastmath=True)
def feed_forward_jit(data_arr, p, Wxh, Whh, bh, Why, by, vocab_size):
    inputs = data_arr[p: p + seq_len]
    n = len(inputs)
    h = np.zeros((n, hidden_size))
    y = np.zeros((n, vocab_size))
    p = np.zeros((n, vocab_size))
    for t in range(n):
        h[t] = update_ht_jit(Wxh, inputs, Whh, h, bh, t, vocab_size).reshape(hidden_size)
        y[t] = np.dot(Why, h[t]) + by  # unnormalized log probabilities for next chars
        p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))  # probabilities for next chars
    return Wxh, Whh, bh, h, y, p


@njit
def cross_entropy_loss(targets, p, t):
    return -np.log(p[t][targets[t], 0])  # softmax (cross-entropy loss)


@njit
def back_propagation(Wxh, Whh, Why, bh, by, x, h, p, inputs, targets):
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(h[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(p[t])
        # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, h[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dhraw = (1 - h[t] * h[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, x[t].T)
        dWhh += np.dot(dhraw, h[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)


class RNN(TextProcess):
    def __init__(self, data):
        super().__init__(data)
        # hyper-parameters
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.learning_rate = learning_rate

        # weights
        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(self.vocab_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros(self.vocab_size)  # output bias

    def feed_forward(self, p):
        return feed_forward_jit(self.data_arr, p, self.Wxh, self.Whh, self.bh, self.Why, self.by, self.vocab_size)
