import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

import data
from models.select_model import SelectModel
import utils
from engine_select import SelectEngine

import pdb

def get_result(model, dataset, sel_crit, top_k=1):
    total_loss = 0
    total_select = 0
    correct_idxs = []
    for batch in dataset:
        sel_out, sel_tgt = SelectEngine.forward(model, batch)
        total_loss += sel_crit(sel_out, sel_tgt).item()
        _, idxs = torch.topk(sel_out, top_k, dim=1)
        correct_idxs += [sel_tgt[i] in idxs[i] for i in range(len(sel_tgt))]
        total_select += sel_out.size(0)
    return total_loss / len(dataset), sum(correct_idxs) / total_select, np.array(correct_idxs)

def main():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data', type=str, default='data/onecommon',
        help='location of the data corpus')
    parser.add_argument('--nembed_word', type=int, default=128,
        help='size of word embeddings')
    parser.add_argument('--nembed_ctx', type=int, default=128,
        help='size of context embeddings')
    parser.add_argument('--nhid_lang', type=int, default=128,
        help='size of the hidden state for the language module')
    parser.add_argument('--nhid_sel', type=int, default=128,
        help='size of the hidden state for the selection module')
    parser.add_argument('--lr', type=float, default=0.001,
        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.1,
        help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.5,
        help='dropout rate in embedding layer')
    parser.add_argument('--init_range', type=float, default=0.01,
        help='initialization range')
    parser.add_argument('--max_epoch', type=int, default=40,
        help='max number of epochs')
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--unk_threshold', type=int, default=10,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--seed', type=int, default=None,
        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--model_file', type=str,  default='tmp.th',
        help='path to save the final model')
    parser.add_argument('--domain', type=str, default='one_common',
        help='domain for the dialogue')
    parser.add_argument('--rel_ctx_encoder', action='store_true', default=False,
        help='wheather to use relational module for encoding the context')
    parser.add_argument('--rel_hidden', type=int, default=128,
        help='size of relation module embeddings')
    parser.add_argument('--context_only', action='store_true', default=False,
        help='train without dialogue embeddings')
    parser.add_argument('--test_corpus', choices=['full', 'uncorrelated', 'success_only'], default='full',
        help='type of test corpus to use')
    parser.add_argument('--test_only', action='store_true', default=False,
        help='use pretrained model for testing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if args.seed is None:
        print("running experiments with 10 different seeds")
        args.seed = list(range(10))
    else:
        args.seed = [args.seed]
    best_valid_loss = 1e8
    best_model = None

    train_accuracies = np.array([])
    valid_accuracies = np.array([])
    test_accuracies = np.array([])
    test_correct = defaultdict(list)

    for seed in args.seed:
        utils.set_seed(seed)

        # consider double count
        freq_cutoff = args.unk_threshold * 2
        corpus = data.WordCorpus(args.data, freq_cutoff=freq_cutoff, verbose=True)
        if args.test_corpus == 'full':
            test_corpus = corpus
        elif args.test_corpus == 'uncorrelated':
            test_corpus = data.WordCorpus(args.data, train='train_uncorrelated.txt', valid='valid_uncorrelated.txt', test='test_uncorrelated.txt',
                freq_cutoff=freq_cutoff, verbose=True, word_dict=corpus.word_dict)
        elif args.test_corpus == 'success_only':
            test_corpus = data.WordCorpus(args.data, train='train_success_only.txt', valid='valid_success_only.txt', test='test_success_only.txt',
                freq_cutoff=freq_cutoff, verbose=True, word_dict=corpus.word_dict)
        
        if args.test_only:
            model = utils.load_model(args.model_file)
        else:
            model = SelectModel(corpus.word_dict, corpus.output_length, args, device)
            if torch.cuda.is_available():
                model = model.to(device)
            engine_select = SelectEngine(model, args, device, verbose=True)
            train_loss, best_valid_loss, best_model_state = engine_select.train(corpus)
            print('best valid loss %.3f' % np.exp(best_valid_loss))
            model.load_state_dict(best_model_state)

        # Test Target Selection
        model.eval()
        
        sel_crit = nn.CrossEntropyLoss()

        trainset, trainset_stats = test_corpus.train_dataset(args.bsz, device=device)
        train_loss, train_accuracy, _ = get_result(model, trainset, sel_crit)

        validset, validset_stats = test_corpus.valid_dataset(args.bsz, device=device)
        valid_loss, valid_accuracy, _ = get_result(model, validset, sel_crit)

        testset, testset_stats = test_corpus.test_dataset(args.bsz, device=device)
        test_loss, test_accuracy, correct_idxs = get_result(model, testset, sel_crit)

        if best_model is None or valid_loss < best_valid_loss:
            best_model = model
            best_valid_loss = valid_loss
            utils.save_model(best_model, args.model_file)
        
        print('trainloss %.5f' % (train_loss))
        print('trainaccuracy {:.5f}'.format(train_accuracy))
        print('validloss %.5f' % (valid_loss))
        print('validaccuracy {:.5f}'.format(valid_accuracy))
        print('testloss %.5f' % (test_loss))
        print('testaccuracy {:.5f}'.format(test_accuracy))

        train_accuracies = np.append(train_accuracies, train_accuracy)
        valid_accuracies = np.append(valid_accuracies, valid_accuracy)
        test_accuracies = np.append(test_accuracies, test_accuracy)

    # print final results
    output = '{:.2f} \\pm {:.1f}'.format(np.mean(train_accuracies) * 100, np.std(train_accuracies) * 100)
    output += ' & {:.2f} \\pm {:.1f}'.format(np.mean(valid_accuracies) * 100, np.std(valid_accuracies) * 100)
    output += ' & {:.2f} \\pm {:.1f}'.format(np.mean(test_accuracies) * 100, np.std(test_accuracies) * 100)
    print(output)

if __name__ == '__main__':
    main()
