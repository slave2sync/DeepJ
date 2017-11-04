import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
from util import *
import numpy as np

class DeepJG(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_actions=NUM_ACTIONS, num_units=512, num_layers=3, style_units=32):
        super().__init__()
        self.num_units = num_units
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.style_units = style_units

        # Project input into distributed representation
        self.input_linear = nn.Linear(num_actions, self.num_units)
        # Project style into distributed representation
        self.style_linear = nn.Linear(NUM_STYLES, self.style_units)
        # Output layer
        self.output_linear = nn.Linear(self.num_units, num_actions + 1)

        self.layers = [RNNLayer(self.num_units, self.num_units) for i in range(num_layers)]

        for i, layer in enumerate(self.layers):
            self.add_module('rnn_layer_' + str(i), layer)

    def forward(self, x, style, states=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Distributed input representation
        x = F.tanh(self.input_linear(x))
        # Distributed style representation
        # style = F.tanh(self.style_linear(style))

        # Initialize state
        if states is None:
            states = [None for _ in range(self.num_layers)]

        for l, (layer, state) in enumerate(zip(self.layers, states)):
            x, states[l] = layer(x, style, state)

        x = self.output_linear(x)

        # Split into value and policy outputs
        value = x[:, :, 0]
        policy = x[:, :, 1:]
        return value, policy, states

    def generate(self, x, style, states, temperature=1):
        """ Returns the probability of outputs """
        _, x, states = self.forward(x, style, states)
        seq_len = x.size(1)
        x = x.view(-1, self.num_actions)
        x = F.softmax(x / temperature)
        x = x.view(-1, seq_len, self.num_actions)
        return x, states

class DeepJD(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_actions=NUM_ACTIONS, num_units=256, num_layers=2, style_units=32):
        super().__init__()
        self.num_units = num_units
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.style_units = style_units

        # Project input into distributed representation
        self.input_linear = nn.Linear(num_actions, self.num_units)
        # Project style into distributed representation
        self.style_linear = nn.Linear(NUM_STYLES, self.style_units)
        # Output layer
        self.output_linear = nn.Linear(self.num_units, 1)

        self.layers = [RNNLayer(self.num_units, self.num_units) for i in range(num_layers)]

        for i, layer in enumerate(self.layers):
            self.add_module('rnn_layer_' + str(i), layer)

    def forward(self, x, style, states=None):
        # Distributed input representation
        x = F.tanh(self.input_linear(x))
        # Distributed style representation
        # style = F.tanh(self.style_linear(style))

        # Initialize state
        if states is None:
            states = [None for _ in range(self.num_layers)]

        for l, (layer, state) in enumerate(zip(self.layers, states)):
            x, states[l] = layer(x, style, state)

        # Only consider last time step
        x = x[:, -1, :]
        x = self.output_linear(x)
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