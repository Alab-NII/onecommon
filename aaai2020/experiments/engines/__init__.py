import argparse
import random
import time
import itertools
import sys
import copy
import re
import os
import shutil

import pdb

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import numpy as np

from keyword_dict import keyword_dict

from logger import Logger


class Criterion(object):
    """Weighted CrossEntropyLoss."""
    def __init__(self, dictionary, device_id=None, bad_toks=[], reduction='mean'):
        w = torch.Tensor(len(dictionary)).fill_(1)
        for tok in bad_toks:
            w[dictionary.get_idx(tok)] = 0.0
        if device_id is not None:
            w = w.cuda(device_id)
        # https://pytorch.org/docs/stable/nn.html
        self.crit = nn.CrossEntropyLoss(w, reduction=reduction)

    def __call__(self, out, tgt):
        return self.crit(out, tgt)

class NormKLLoss(_Loss):
    def __init__(self, unit_average=False):
        super(NormKLLoss, self).__init__()
        self.unit_average = unit_average

    def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        # find the KL divergence between two Gaussian distribution
        loss = 1.0 + (recog_logvar - prior_logvar)
        loss -= torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
        loss -= torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar))
        if self.unit_average:
            kl_loss = -0.5 * torch.mean(loss, dim=1)
        else:
            kl_loss = -0.5 * torch.sum(loss, dim=1)
        avg_kl_loss = torch.mean(kl_loss)
        return avg_kl_loss

class CatKLLoss(_Loss):
    def __init__(self):
        super(CatKLLoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """
        qy * log(q(y)/p(y))
        """
        qy = torch.exp(log_qy)
        y_kl = torch.sum(qy * (log_qy - log_py), dim=1)
        if unit_average:
            return torch.mean(y_kl)
        else:
            return torch.sum(y_kl)/batch_size

class Entropy(_Loss):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, log_qy, batch_size=None, unit_average=False):
        """
        -qy log(qy)
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = torch.exp(log_qy)
        h_q = torch.sum(-1 * log_qy * qy, dim=1)
        if unit_average:
            return torch.mean(h_q)
        else:
            return torch.sum(h_q) / batch_size

class EngineBase(object):
    """Base class for training engine."""
    def __init__(self, model, args, verbose=False):
        self.model = model
        self.args = args
        self.verbose = verbose
        self.opt = self.make_opt(self.args.lr)
        self.crit = Criterion(self.model.word_dict)
        self.sel_crit = nn.CrossEntropyLoss(reduction='mean')
        self.ref_crit = nn.BCEWithLogitsLoss(reduction='mean')
        if args.tensorboard_log:
            log_name = 'tensorboard_logs/{}'.format(args.model_type)
            if os.path.exists(log_name):
                print("remove old tensorboard log")
                shutil.rmtree(log_name)
            self.logger = Logger(log_name)

    def make_opt(self, lr):
        if self.args.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr)
        elif self.args.optimizer == 'rmsprop':
            return optim.RMSprop(
            self.model.parameters(),
            lr=lr,
            momentum=self.args.momentum)
        else:
            assert False

    def get_model(self):
        return self.model

    def train_batch(self, batch):
        pass

    def valid_batch(self, batch):
        pass

    def train_pass(self, trainset, trainset_stats):
        '''
        basic implementation of one training pass
        '''
        self.model.train()

        total_lang_loss, total_select_loss, total_num_correct, total_num_select = 0, 0, 0, 0
        start_time = time.time()

        for batch in trainset:
            lang_loss, select_loss, num_correct, num_select = self.train_batch(batch)
            total_lang_loss += lang_loss
            total_select_loss += select_loss
            total_num_correct += num_correct
            total_num_select += num_select

        total_lang_loss /= len(trainset)
        total_select_loss /= len(trainset)
        time_elapsed = time.time() - start_time
        return total_lang_loss, total_select_loss, total_num_correct / total_num_select, time_elapsed

    def valid_pass(self, validset, validset_stats):
        '''
        basic implementation of one validation pass
        '''
        self.model.eval()

        total_lang_loss, total_select_loss, total_num_correct, total_num_select = 0, 0, 0, 0
        for batch in validset:
            lang_loss, select_loss, num_correct, num_select = self.valid_batch(batch)
            total_lang_loss += lang_loss
            total_select_loss += select_loss
            total_num_correct += num_correct
            total_num_select += num_select

        total_lang_loss /= len(validset)
        total_select_loss /= len(validset)
        return total_lang_loss, total_select_loss, total_num_correct / total_num_select

    def iter(self, epoch, lr, traindata, validdata):
        trainset, trainset_stats = traindata
        validset, validset_stats = validdata

        train_lang_loss, train_select_loss, train_select_accuracy, train_time = self.train_pass(trainset, trainset_stats)
        valid_lang_loss, valid_select_loss, valid_select_accuracy = self.valid_pass(validset, validset_stats)

        if self.verbose:
            print('| epoch %03d | trainlangloss %.6f | trainlangppl %.6f | s/epoch %.2f | lr %0.8f' % (
                epoch, train_lang_loss, np.exp(train_lang_loss), train_time, lr))
            print('| epoch %03d | trainselectloss(scaled) %.6f | trainselectaccuracy %.4f | s/epoch %.2f | lr %0.8f' % (
                epoch, train_select_loss * self.args.sel_weight, train_select_accuracy, train_time, lr))
            print('| epoch %03d | validlangloss %.6f | validlangppl %.8f' % (
                epoch, valid_lang_loss, np.exp(valid_lang_loss)))
            print('| epoch %03d | validselectloss %.6f | validselectaccuracy %.4f' % (
                epoch, valid_select_loss, valid_select_accuracy))

        if self.args.tensorboard_log:
            info = {'Train_Lang_Loss': train_lang_loss,
                'Train_Select_Loss': train_select_loss,
                'Valid_Lang_Loss': valid_lang_loss,
                'Valid_Select_Loss': valid_select_loss,
                'Valid_Select_Accuracy': valid_select_accuracy}
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, epoch)

            for tag, value in self.model.named_parameters():
                if value.grad is None:
                    continue
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                self.logger.histo_summary(
                    tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        return valid_lang_loss, valid_select_loss

    def combine_loss(self, lang_loss, select_loss):
        return lang_loss + select_loss * self.args.sel_weight

    def train(self, corpus):
        best_model, best_combined_valid_loss = copy.deepcopy(self.model), 1e100
        validdata = corpus.valid_dataset(self.args.bsz)
        
        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz)
            valid_lang_loss, valid_select_loss = self.iter(epoch, self.args.lr, traindata, validdata)

            combined_valid_loss = self.combine_loss(valid_lang_loss, valid_select_loss)
            if combined_valid_loss < best_combined_valid_loss:
                print("update best model: validlangloss %.8f | validselectloss %.8f" % 
                    (valid_lang_loss, valid_select_loss))
                best_combined_valid_loss = combined_valid_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

        return best_combined_valid_loss, best_model

    def train_scheduled(self, corpus):
        best_model, best_combined_valid_loss = copy.deepcopy(self.model), 1e100
        lr = self.args.lr
        last_decay_epoch = 0
        self.t = 0
        validdata = corpus.valid_dataset(self.args.bsz)

        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz)
            valid_lang_loss, valid_select_loss = self.iter(epoch, lr, traindata, validdata)

            combined_valid_loss = self.combine_loss(valid_lang_loss, valid_select_loss)
            if combined_valid_loss < best_combined_valid_loss:
                print("update best model: validlangloss %.8f | validselectloss %.8f" % 
                    (valid_lang_loss, valid_select_loss))
                best_combined_valid_loss = combined_valid_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

        if self.verbose:
            print('| start annealing | best combined loss %.3f | best combined ppl %.3f' % (
                best_combined_valid_loss, np.exp(best_combined_valid_loss)))

        self.model = best_model
        for epoch in range(self.args.max_epoch + 1, 100):
            if epoch - last_decay_epoch >= self.args.decay_every:
                last_decay_epoch = epoch
                lr /= self.args.decay_rate
                if lr < self.args.min_lr:
                    break
                self.opt = self.make_opt(lr)

            traindata = corpus.train_dataset(self.args.bsz)
            valid_lang_loss, valid_select_loss = self.iter(
                epoch, lr, traindata, validdata)

            combined_valid_loss = self.combine_loss(valid_lang_loss, valid_select_loss)
            if combined_valid_loss < best_combined_valid_loss:
                print("update best model: validlangloss %.8f | validselectloss %.8f" % 
                    (valid_lang_loss, valid_select_loss))
                best_combined_valid_loss = combined_valid_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

        return best_combined_valid_loss, best_model

