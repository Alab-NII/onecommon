from itertools import combinations
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F


def init_rnn(rnn, init_range, weights=None, biases=None, bidirectional=False):
    """Initializes RNN uniformly."""
    weights = weights or ['weight_ih_l0', 'weight_hh_l0']
    biases = biases or ['bias_ih_l0', 'bias_hh_l0']
    # Init weights
    for w in weights:
        rnn._parameters[w].data.uniform_(-init_range, init_range)
    # Init biases
    for b in biases:
        rnn._parameters[b].data.fill_(0)
    if bidirectional:
        reverse_weights = ['weight_ih_l0_reverse', 'weight_hh_l0_reverse']
        reverse_biases = ['bias_ih_l0_reverse', 'bias_hh_l0_reverse']
        for w in reverse_weights:
            rnn._parameters[w].data.uniform_(-init_range, init_range)
        for b in reverse_biases:
            rnn._parameters[b].data.fill_(0)

def init_rnn_cell(rnn, init_range):
    """Initializes RNNCell uniformly."""
    init_rnn(rnn, init_range, ['weight_ih', 'weight_hh'], ['bias_ih', 'bias_hh'])


def init_cont(cont, init_range):
    """Initializes a container uniformly."""
    for m in cont:
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-init_range, init_range)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0)


class RelationalContextEncoder(nn.Module):
    """A module that encodes dialogues context using an MLP."""
    def __init__(self, num_ent, dim_ent, rel_hidden, hidden_size, dropout, init_range, device):
        super(RelationalContextEncoder, self).__init__()

        self.input_size = num_ent * dim_ent + rel_hidden
        self.num_ent = num_ent
        self.dim_ent = dim_ent
        self.hidden_size = hidden_size
        self.device = device
        self.rel_hidden = rel_hidden

        self.relation = nn.Sequential(
            torch.nn.Linear(2 * self.dim_ent, self.rel_hidden),
            nn.Tanh()
        )

        self.fc1 = nn.Linear(self.input_size, self.hidden_size) 
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        init_cont([self.fc1, self.relation], init_range)

    def forward(self, ctx):
        ctx.to(self.device)
        ents = ctx.view(ctx.size(0), self.num_ent, self.dim_ent)
        rel = torch.cat([self.relation(torch.cat([ents[:,i,:],ents[:,j,:]], 1)) for i, j in combinations(range(7), 2)], 1)
        rel = torch.sum(rel.view(rel.size(0), self.rel_hidden, -1), 2)

        inpt = torch.cat([ctx, rel], 1)
        out = self.dropout(inpt)
        out = self.fc1(inpt)
        out = self.tanh(out)
        return out.unsqueeze(0)


class MlpContextEncoder(nn.Module):
    """A module that encodes dialogues context using an MLP."""
    def __init__(self, input_size, hidden_size, dropout, init_range, device):
        super(MlpContextEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device

        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        init_cont([self.fc1], init_range)

    def forward(self, ctx):
        ctx.to(self.device)
        out = self.fc1(ctx)
        out = self.dropout(out)
        out = self.tanh(out)
        return out.unsqueeze(0)
