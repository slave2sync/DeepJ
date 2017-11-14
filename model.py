import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

from constants import *
from util import *
import numpy as np
import math

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_units=512, num_layers=3, style_units=32):
        super().__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.style_units = style_units

        # RNN
        self.rnns = [GRUCell(NUM_ACTIONS + style_units if i == 0 else self.num_units, self.num_units) for i in range(num_layers)]
        # self.rnn = nn.LSTM(NUM_ACTIONS + style_units, self.num_units, num_layers, batch_first=True)

        self.output_linear = nn.Linear(self.num_units, NUM_ACTIONS)

        for i, rnn in enumerate(self.rnns):
            self.add_module('rnn_' + str(i), rnn)

        # Style
        self.style_linear = nn.Linear(NUM_STYLES, self.style_units)
        # self.style_layer = nn.Linear(self.style_units, self.num_units * self.num_layers)

    def forward(self, x, style, states=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Distributed style representation
        style = self.style_linear(style)
        # style = F.tanh(self.style_layer(style))
        style = style.unsqueeze(1).expand(batch_size, seq_len, self.style_units)
        x = torch.cat((x, style), dim=2)

        ## Process RNN ##
        if states is None:
            states = [None for _ in range(self.num_layers)]

        for l, rnn in enumerate(self.rnns):
            all_h = []
            for t in range(seq_len):
                h = rnn(x[:, t], states[l])
                states[l] = h
                all_h.append(h)
                # Style integration
                # x = x + style[:, l * self.num_units:(l + 1) * self.num_units].unsqueeze(1).expand(-1, seq_len, -1)
        
            x = torch.stack(all_h, dim=1)

        x = self.output_linear(x)
        return x, states

    def generate(self, x, style, states, temperature=1):
        """ Returns the probability of outputs """
        x, states = self.forward(x, style, states)
        seq_len = x.size(1)
        x = x.view(-1, NUM_ACTIONS)
        x = F.softmax(x / temperature)
        x = x.view(-1, seq_len, NUM_ACTIONS)
        return x, states

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.w_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.w_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.b_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.b_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)
        self.reset_parameters()

        self.l1 = LayerNorm(3 * hidden_size)
        self.l2 = LayerNorm(3 * hidden_size)
        self.l3 = LayerNorm(hidden_size)
        self.l4 = LayerNorm(hidden_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):
        if hidden is None:
            hidden = var(torch.zeros(1, self.hidden_size))

        gi = self.l1(F.linear(input, self.w_ih, self.b_ih))
        gh = self.l2(F.linear(hidden, self.w_hh, self.b_hh))
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(self.l3(i_n) + resetgate * self.l4(h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta