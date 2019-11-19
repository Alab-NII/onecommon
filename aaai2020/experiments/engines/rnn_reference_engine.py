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

from keyword_dict import keyword_dict

from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.nn.modules.loss import _Loss
import numpy as np

from engines import EngineBase, Criterion
import utils


class RnnReferenceEngine(EngineBase):
    def __init__(self, model, args, verbose=False):
        super(RnnReferenceEngine, self).__init__(model, args, verbose)

    def _forward(self, batch):
        ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt, scenario_ids, _, _, _, sel_idx = batch

        ctx = Variable(ctx)
        inpt = Variable(inpt)
        if ref_inpt is not None:
            ref_inpt = Variable(ref_inpt)
        out, ref_out, sel_out = self.model.forward(ctx, inpt, ref_inpt, sel_idx)

        tgt = Variable(tgt)
        sel_tgt = Variable(sel_tgt)
        lang_loss = self.crit(out, tgt)

        if ref_inpt is not None:
            ref_tgt = Variable(ref_tgt)
            ref_tgt = torch.transpose(ref_tgt, 0, 1).contiguous().float()
            ref_loss = self.ref_crit(ref_out, ref_tgt)
            ref_correct = ((ref_out > 0).long() == ref_tgt.long()).sum().item()
            ref_total = ref_tgt.size(0) * ref_tgt.size(1) * ref_tgt.size(2)
        else:
            ref_loss = None
            ref_correct = 0
            ref_total = 0

        sel_loss = self.sel_crit(sel_out, sel_tgt)
        sel_correct = (sel_out.max(dim=1)[1] == sel_tgt).sum().item()
        sel_total = sel_out.size(0)

        return lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total

    def train_batch(self, batch):
        lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total = self._forward(batch)

        # default
        loss = None
        return_ref_loss = 0
        if self.args.lang_weight > 0:
            loss = self.args.lang_weight * lang_loss
            if self.args.sel_weight > 0:
                loss += self.args.sel_weight * sel_loss
            if self.args.ref_weight > 0 and ref_loss is not None:
                loss += self.args.ref_weight * ref_loss
                return_ref_loss = ref_loss.item()
        elif self.args.sel_weight > 0:
            loss = self.args.sel_weight * sel_loss / sel_total
            if self.args.ref_weight > 0 and ref_loss is not None:
                loss += self.args.ref_weight * ref_loss
                return_ref_loss = ref_loss.item()
        elif self.args.ref_weight > 0 and ref_loss is not None:
            loss = self.args.ref_weight * ref_loss
            return_ref_loss = ref_loss.item()

        if loss:
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.opt.step()

        return lang_loss.item(), return_ref_loss, ref_correct, ref_total, sel_loss.item(), sel_correct, sel_total

    def valid_batch(self, batch):
        with torch.no_grad():
            lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total = self._forward(batch)

        if ref_loss is not None:
            return_ref_loss = ref_loss.item()
        else:
            return_ref_loss = 0

        return lang_loss.item(), return_ref_loss, ref_correct, ref_total, sel_loss.item(), sel_correct, sel_total

    def test_batch(self, batch):
        with torch.no_grad():
            lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total = self._forward(batch)

        if ref_loss is not None:
            return_ref_loss = ref_loss.item()
        else:
            return_ref_loss = 0

        return lang_loss.item(), return_ref_loss, ref_correct, ref_total, sel_loss.item(), sel_correct, sel_total

    def train_pass(self, trainset, trainset_stats):
        '''
        basic implementation of one training pass
        '''
        self.model.train()

        total_lang_loss, total_select_loss, total_select, total_select_correct, total_reference_loss, total_reference, total_reference_correct = 0, 0, 0, 0, 0, 0, 0
        start_time = time.time()

        for batch in trainset:
            lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total = self.train_batch(batch)
            total_lang_loss += lang_loss
            total_select_loss += sel_loss
            total_select_correct += sel_correct
            total_select += sel_total
            total_reference_loss += ref_loss
            total_reference_correct += ref_correct
            total_reference += ref_total

        total_lang_loss /= len(trainset)
        total_select_loss /= len(trainset)
        total_reference_loss /= len(trainset)
        time_elapsed = time.time() - start_time
        return total_lang_loss, total_reference_loss, total_reference_correct / total_reference, total_select_loss, total_select_correct / total_select, time_elapsed

    def valid_pass(self, validset, validset_stats):
        '''
        basic implementation of one validation pass
        '''
        self.model.eval()

        total_lang_loss, total_select_loss, total_select, total_select_correct, total_reference_loss, total_reference, total_reference_correct = 0, 0, 0, 0, 0, 0, 0
        for batch in validset:
            lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total = self.valid_batch(batch)
            total_lang_loss += lang_loss
            total_select_loss += sel_loss
            total_select_correct += sel_correct
            total_select += sel_total
            total_reference_loss += ref_loss
            total_reference_correct += ref_correct
            total_reference += ref_total

        total_lang_loss /= len(validset)
        total_select_loss /= len(validset)
        total_reference_loss /= len(validset)
        return total_lang_loss, total_reference_loss, total_reference_correct / total_reference, total_select_loss, total_select_correct / total_select

    def iter(self, epoch, lr, traindata, validdata):
        trainset, trainset_stats = traindata
        validset, validset_stats = validdata

        train_lang_loss, train_reference_loss, train_reference_accuracy, train_select_loss, train_select_accuracy, train_time = self.train_pass(trainset, trainset_stats)
        valid_lang_loss, valid_reference_loss, valid_reference_accuracy, valid_select_loss, valid_select_accuracy = self.valid_pass(validset, validset_stats)

        if self.verbose:
            print('| epoch %03d | trainlangloss(scaled) %.6f | trainlangppl %.6f | s/epoch %.2f | lr %0.8f' % (
                epoch, train_lang_loss * self.args.lang_weight, np.exp(train_lang_loss), train_time, lr))
            print('| epoch %03d | trainselectloss(scaled) %.6f | trainselectaccuracy %.4f | s/epoch %.2f | lr %0.8f' % (
                epoch, train_select_loss * self.args.sel_weight, train_select_accuracy, train_time, lr))
            print('| epoch %03d | trainreferenceloss(scaled) %.6f | trainreferenceaccuracy %.4f | s/epoch %.2f | lr %0.8f' % (
                epoch, train_reference_loss * self.args.ref_weight, train_reference_accuracy, train_time, lr))
            print('| epoch %03d | validlangloss %.6f | validlangppl %.8f' % (
                epoch, valid_lang_loss, np.exp(valid_lang_loss)))
            print('| epoch %03d | validselectloss %.6f | validselectaccuracy %.4f' % (
                epoch, valid_select_loss, valid_select_accuracy))
            print('| epoch %03d | validreferenceloss %.6f | validreferenceaccuracy %.4f' % (
                epoch, valid_reference_loss, valid_reference_accuracy))

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

        return valid_lang_loss, valid_select_loss, valid_reference_loss

    def combine_loss(self, lang_loss, select_loss, reference_loss):
        return lang_loss * int(self.args.lang_weight > 0) + select_loss * int(self.args.sel_weight > 0) + reference_loss * int(self.args.ref_weight > 0)

    def train(self, corpus):
        best_model, best_combined_valid_loss = copy.deepcopy(self.model), 1e100
        validdata = corpus.valid_dataset(self.args.bsz)
        
        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz)
            valid_lang_loss, valid_select_loss, valid_reference_loss = self.iter(epoch, self.args.lr, traindata, validdata)

            combined_valid_loss = self.combine_loss(valid_lang_loss, valid_select_loss, valid_reference_loss)
            if combined_valid_loss < best_combined_valid_loss:
                print("update best model: validlangloss %.8f | validselectloss %.8f | validreferenceloss %.8f" % 
                    (valid_lang_loss, valid_select_loss, valid_reference_loss))
                best_combined_valid_loss = combined_valid_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

        return best_combined_valid_loss, best_model

