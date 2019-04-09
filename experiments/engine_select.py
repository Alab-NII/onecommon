import argparse
import random
import pdb
import time
import itertools
import sys
import copy
import re

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from data import STOP_TOKENS

from logger import Logger

class SelectEngine(object):
    """The training engine.

    Performs training and evaluation.
    """
    def __init__(self, model, args, device=None, verbose=False):
        self.model = model
        self.args = args
        self.device = device
        self.verbose = verbose
        self.opt = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        self.sel_crit = nn.CrossEntropyLoss()
        self.logger = Logger('tensorboard_logs_{}'.format(args.model_file))

    def forward(model, batch, requires_grad=False):
        """A helper function to perform a forward pass on a batch."""
        # extract the batch into contxt, input, target and selection target
        with torch.set_grad_enabled(requires_grad):
            ctx, inpt, tgt, sel_tgt = batch
            # create variables
            ctx = Variable(ctx)
            inpt = Variable(inpt)
            tgt = Variable(tgt)
            sel_tgt = Variable(sel_tgt)

            ctx_h = model.forward_context(ctx)
            lang_h = model.zero_hid(ctx_h.size(0), model.args.nhid_lang)
            sel_out = model.forward_selection(inpt, lang_h, ctx_h)

            return sel_out, sel_tgt

    def train_pass(self, N, trainset):
        """Training pass."""
        self.model.train()

        total_loss = 0
        total_correct = 0
        start_time = time.time()

        for batch in trainset:
            sel_out, sel_tgt = SelectEngine.forward(self.model, batch, requires_grad=True)
            loss = self.sel_crit(sel_out, sel_tgt)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.opt.step()

            total_loss += loss.item()
            total_correct += (sel_out.max(dim=1)[1] == sel_tgt).sum().item()

        total_loss /= len(trainset)
        time_elapsed = time.time() - start_time
        return total_loss, time_elapsed, total_correct / (len(trainset) * self.args.bsz)

    def valid_pass(self, N, validset):
        """Validation pass."""
        self.model.eval()

        total_loss = 0
        total_correct = 0

        for batch in validset:
            sel_out, sel_tgt = SelectEngine.forward(self.model, batch, requires_grad=False)
            loss = self.sel_crit(sel_out, sel_tgt)

            total_loss += loss.item()
            total_correct += (sel_out.max(dim=1)[1] == sel_tgt).sum().item()

        return  total_loss / len(validset), total_correct / (len(validset) * self.args.bsz)

    def iter(self, N, epoch, lr, traindata, validdata):
        """Performs on iteration of the training.
        Runs one epoch on the training and validation datasets.
        """
        trainset, _ = traindata
        validset, _ = validdata

        train_loss, train_time, train_accuracy = self.train_pass(N, trainset)
        valid_loss, valid_accuracy = self.valid_pass(N, validset)

        if self.verbose:
            print('| epoch %03d | trainloss %.3f | s/epoch %.2f | trainaccuracy %.3f | lr %0.8f' % (
                epoch, train_loss, train_time, train_accuracy, lr))
            print('| epoch %03d | validloss %.3f | validselectppl %.3f' % (
                epoch, valid_loss, np.exp(valid_loss)))
            print('| epoch %03d | valid_select_accuracy %.3f' % (
                epoch, valid_accuracy))

        # Tensorboard Logging
        # 1. Log scalar values (scalar summary)
        info = {'Train_Loss': train_loss,
                'Train_Accuracy': train_accuracy,
                'Valid_Loss': valid_loss,
                'Valid_Accuracy': valid_accuracy}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.model.named_parameters():
            if value.grad is None:
                # don't need to log untrainable parameters
                continue
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            self.logger.histo_summary(
                tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        return train_loss, valid_loss

    def train(self, corpus):
        """Entry point."""
        N = len(corpus.word_dict)
        best_model, best_valid_loss = None, 1e100
        lr = self.args.lr

        validdata = corpus.valid_dataset(self.args.bsz, device=self.device)
        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz, device=self.device)
            train_loss, valid_loss = self.iter(N, epoch, lr, traindata, validdata)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_state = copy.deepcopy(self.model.state_dict())

        return train_loss, best_valid_loss, best_model_state
