import argparse
import sys
import time
import random
import itertools
import re
import pdb
import copy

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import data
import models
from models.markable_detector import BiLSTM_CRF
import utils
from domain import get_domain

START_TAG = "<START>"
STOP_TAG = "<STOP>"

def main():
    parser = argparse.ArgumentParser(description='training script for markable detection')
    parser.add_argument('--data', type=str, default='data/onecommon',
        help='location of the data corpus')
    parser.add_argument('--nembed_word', type=int, default=128,
        help='size of word embeddings')
    parser.add_argument('--nembed_ctx', type=int, default=128,
        help='size of context embeddings')
    parser.add_argument('--nhid_lang', type=int, default=128,
        help='size of the hidden state for the language module')
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
    parser.add_argument('--max_epoch', type=int, default=10,
        help='max number of epochs')
    parser.add_argument('--bsz', type=int, default=1,
        help='batch size')
    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--model_file', type=str,  default='markable_detector',
        help='path to save the final model')
    parser.add_argument('--domain', type=str, default='one_common',
        help='domain for the dialogue')
    parser.add_argument('--tensorboard_log', action='store_true', default=False,
        help='log training with tensorboard')
    parser.add_argument('--repeat_train', action='store_true', default=False,
        help='repeat training n times')
    parser.add_argument('--test_only', action='store_true', default=False,
        help='test only')
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

        corpus = BiLSTM_CRF.corpus_ty(domain, args.data, train='train_markable_{}.txt'.format(seed), valid='valid_markable_{}.txt'.format(seed), test='test_markable_{}.txt'.format(seed), verbose=True)

        if args.test_only:
            best_model = utils.load_model(args.model_file + '_' + str(seed) + '.th')
            if args.cuda:
                best_model.cuda()
            else:
                device = torch.device("cpu")
                best_model.to(device)
            best_model.eval()
        else:
            model = BiLSTM_CRF(len(corpus.word_dict), corpus.bio_dict, args.nembed_word, args.nhid_lang)
            optimizer = optim.Adam(
                    model.parameters(),
                    lr=args.lr)

            if args.cuda:
                model.cuda()

            best_model, best_valid_loss = copy.deepcopy(model), 1e100
            validdata = corpus.valid_dataset(args.bsz)
            
            for epoch in range(1, args.max_epoch + 1):
                traindata = corpus.train_dataset(args.bsz)

                trainset, trainset_stats = traindata
                validset, validset_stats = validdata

                # train pass
                model.train()
                total_lang_loss, total_select_loss, total_num_correct, total_num_select = 0, 0, 0, 0
                start_time = time.time()

                for batch in tqdm(trainset):
                    model.zero_grad()

                    ctx, words, markables, scenario_ids, agents, chat_ids = batch

                    ctx = Variable(ctx)
                    words = Variable(words)
                    markables = Variable(markables)

                    loss = model.neg_log_likelihood(words, markables)
                    loss.sum().backward()
                    optimizer.step()

                # valid pass
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    valid_loss = 0
                    for batch in tqdm(validset):
                        ctx, words, markables, scenario_ids, agents, chat_ids = batch

                        valid_loss += model.neg_log_likelihood(words, markables).sum().item()

                        score, tag_seq = model(words)
                        correct += (torch.Tensor(tag_seq).long() == markables).sum().item()
                        total += len(tag_seq)

                    print("epoch {}".format(epoch))
                    print("valid loss: {:.5f}".format(valid_loss))
                    print("valid accuracy: {:.5f}".format(correct / total))

                    if valid_loss < best_valid_loss:
                        print("update best model")
                        best_model = copy.deepcopy(model)
                        best_valid_loss = valid_loss

        # test pass
        testdata = corpus.test_dataset(args.bsz)
        testset, testset_stats = testdata
        best_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0
            for batch in tqdm(testset):
                ctx, words, markables, scenario_ids, agents, chat_ids = batch

                test_loss += best_model.neg_log_likelihood(words, markables).sum().item()

                score, tag_seq = best_model(words)
                correct += (torch.Tensor(tag_seq).long() == markables).sum().item()
                total += len(tag_seq)

            print("final test {}".format(epoch))
            print("test loss: {:.5f}".format(test_loss))
            print("test accuracy: {:.5f}".format(correct / total))

        if not args.test_only:
            utils.save_model(best_model, args.model_file + '_' + str(seed) + '.th')
            utils.save_model(best_model.state_dict(), 'stdict_' + args.model_file)

if __name__ == '__main__':
    main()
