import argparse
import sys
import time
import random
import itertools
import re
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import data
import models
import utils
from domain import get_domain


def main():
    parser = argparse.ArgumentParser(description='training script for reference resolution')
    parser.add_argument('--data', type=str, default='data/onecommon',
        help='location of the data corpus')
    parser.add_argument('--model_type', type=str, default='rnn_reference_model',
        help='type of model to use', choices=models.get_model_names())
    parser.add_argument('--ctx_encoder_type', type=str, default='mlp_encoder',
        help='type of context encoder to use', choices=models.get_ctx_encoder_names())
    parser.add_argument('--attention', action='store_true', default=False,
        help='use attention')
    parser.add_argument('--nembed_word', type=int, default=128,
        help='size of word embeddings')
    parser.add_argument('--nhid_rel', type=int, default=64,
        help='size of the hidden state for the language module')
    parser.add_argument('--nembed_ctx', type=int, default=128,
        help='size of context embeddings')
    parser.add_argument('--nembed_cond', type=int, default=128,
        help='size of condition embeddings')
    parser.add_argument('--nhid_lang', type=int, default=128,
        help='size of the hidden state for the language module')
    parser.add_argument('--nhid_strat', type=int, default=128,
        help='size of the hidden state for the strategy module')
    parser.add_argument('--nhid_attn', type=int, default=64,
        help='size of the hidden state for the attention module')
    parser.add_argument('--nhid_sel', type=int, default=64,
        help='size of the hidden state for the selection module')
    parser.add_argument('--share_attn', action='store_true', default=False,
        help='share attention modules for selection and language output')
    parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam',
        help='optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001,
        help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
        help='min threshold for learning rate annealing')
    parser.add_argument('--decay_rate', type=float,  default=9.0,
        help='decrease learning rate by this factor')
    parser.add_argument('--decay_every', type=int,  default=1,
        help='decrease learning rate after decay_every epochs')
    parser.add_argument('--momentum', type=float, default=0.0,
        help='momentum for sgd')
    parser.add_argument('--clip', type=float, default=0.5,
        help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.5,
        help='dropout rate in embedding layer')
    parser.add_argument('--init_range', type=float, default=0.01,
        help='initialization range')
    parser.add_argument('--max_epoch', type=int, default=20,
        help='max number of epochs')
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--temperature', type=float, default=0.1,
        help='temperature')
    parser.add_argument('--lang_weight', type=float, default=1.0,
        help='language loss weight')
    parser.add_argument('--ref_weight', type=float, default=1.0,
        help='reference loss weight')
    parser.add_argument('--sel_weight', type=float, default=1.0,
        help='selection loss weight')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--model_file', type=str,  default='tmp.th',
        help='path to save the final model')
    parser.add_argument('--domain', type=str, default='one_common',
        help='domain for the dialogue')
    parser.add_argument('--tensorboard_log', action='store_true', default=False,
        help='log training with tensorboard')
    parser.add_argument('--repeat_train', action='store_true', default=False,
        help='repeat training n times')
    parser.add_argument('--corpus_type', choices=['full', 'uncorrelated', 'success_only'], default='full',
        help='type of training corpus to use')
    args = parser.parse_args()

    if args.repeat_train:
        seeds = list(range(10))
    else:
        seeds = [1]

    for seed in seeds:
        utils.use_cuda(args.cuda)
        utils.set_seed(args.seed)

        domain = get_domain(args.domain)
        model_ty = models.get_model_type(args.model_type)

        corpus = model_ty.corpus_ty(domain, args.data, train='train_reference_{}.txt'.format(seed), valid='valid_reference_{}.txt'.format(seed), test='test_reference_{}.txt'.format(seed),
            freq_cutoff=args.unk_threshold, verbose=True)

        model = model_ty(corpus.word_dict, args)
        if args.cuda:
            model.cuda()

        engine = model_ty.engine_ty(model, args, verbose=True)
        if args.optimizer == 'adam':
            best_valid_loss, best_model = engine.train(corpus)
        elif args.optimizer == 'rmsprop':
            best_valid_loss, best_model = engine.train_scheduled(corpus)

        utils.save_model(best_model, args.model_file + '_' + str(seed) + '.th')
        utils.save_model(best_model.state_dict(), 'stdict_' + args.model_file)


if __name__ == '__main__':
    main()
