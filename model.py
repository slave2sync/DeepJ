import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
from util import *
import numpy as np

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_actions=NUM_ACTIONS, num_units=512, style_units=32):
        super().__init__()
        self.num_units = num_units
        self.num_actions = num_actions
        self.style_units = style_units

        # Project input into distributed representation
        self.input_linear = nn.Linear(num_actions, self.num_units)
        # Project style into distributed representation
        self.style_linear = nn.Linear(NUM_STYLES, self.style_units)
        # Output layer
        self.g_linear = nn.Linear(self.num_units, num_actions + 1)
        self.d_linear = nn.Linear(self.num_units, 1)

        # Shared RNN base
        self.rnn_base = RNNLayer(self.num_units, self.num_units)
        self.rnn_g = [RNNLayer(self.num_units, self.num_units) for i in range(2)]
        self.rnn_d = [RNNLayer(self.num_units, self.num_units)]

        for i, layer in enumerate(self.rnn_g):
            self.add_module('rnn_g_' + str(i), layer)

        for i, layer in enumerate(self.rnn_d):
            self.add_module('rnn_d_' + str(i), layer)

    def forward(self, x, style, states=None, no_g=False, no_d=False):
        # Distributed input representation
        x = F.tanh(self.input_linear(x))
        # Distributed style representation
        # style = F.tanh(self.style_linear(style))

        # Initialize state
        if states is None:
            states = [None for _ in range(4)]

        x, states[0] = self.rnn_base(x, style, states[0])

        if not no_g:
            # Generator
            g = x
            g, states[1] = self.rnn_g[0](g, style, states[1])
            g, states[2] = self.rnn_g[1](g, style, states[2])
            g = self.g_linear(g)

            # Split into value and policy outputs
            value = g[:, :, 0]
            policy = g[:, :, 1:]

        if not no_d:
            # Discriminator
            d = x
            d, states[3] = self.rnn_d[0](d, style, states[3])
            # Discriminator evaluated based on the last time step
            d = d[:, -1, :]
            d = self.d_linear(d)

        if no_g:
            return d
        if no_d:
            return value, policy, states

        return value, policy, d, states

    def generate(self, x, style, states, temperature=1):
        """ Returns the probability of outputs """
        _, _, x, states = self.forward(x, style, states)
        seq_len = x.size(1)
        x = x.view(-1, self.num_actions)
        x = F.softmax(x / temperature)
        x = x.view(-1, seq_len, self.num_actions)
        return x, states

class RNNLayer(nn.Module):
    """
    A DeepJ RNN layer that contains an LSTM and style layer
    """
    def __init__(self, num_inputs, num_units):
        super().__init__()
        self.num_units = num_units
        self.rnn = nn.LSTM(num_inputs, num_units, batch_first=True)
        self.style_layer = nn.Linear(num_inputs, num_units)
    
    def forward(self, x, style, state=None):
        """ Takes input x and style embeddings """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Style integration
        # style_activation = F.tanh(self.style_layer(style))
        # style_seq = style_activation.unsqueeze(1)
        # style_seq = style_seq.expand(batch_size, seq_len, self.num_units)
        # x = x + style_seq

        x, state = self.rnn(x, state)
        return x, state