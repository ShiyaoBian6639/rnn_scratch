from fastrnn.utils.model import RNN
# data I/O
data = open('./input.txt', 'r').read()  # should be simple plain text file
rnn = RNN(data)

Wxh, Whh, bh, h, y, p = rnn.feed_forward(0)
