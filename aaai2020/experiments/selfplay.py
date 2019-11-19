import argparse
import json
import os
import pdb
import re
import random

import numpy as np
import torch
from torch import optim
from torch import autograd
import torch.nn as nn

from agent import *
import utils
from utils import ContextGenerator
from dialog import Dialog, DialogLogger
from models.rnn_reference_model import RnnReferenceModel
import domain

def dump_json(file, path):
    try:
        with open(path, "w") as fout:
            json.dump(file, fout, indent=4, sort_keys=True)
    except:
        raise Exception('Error writing JSON to %s' % path)

class SelfPlay(object):
    def __init__(self, dialog, ctx_gen, args, logger=None):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.logger = logger if logger else DialogLogger()

    def run(self):
        n = 0
        success = 0
        for ctxs in self.ctx_gen.iter():
            n += 1
            if self.args.smart_alice and n > 1000:
                break
            self.logger.dump('=' * 80)
            _, agree, _ = self.dialog.run(ctxs, self.logger)
            if agree:
                success += 1
            self.logger.dump('=' * 80)
            self.logger.dump('')
            if n % 100 == 0:
                self.logger.dump('%d: %s' % (n, self.dialog.show_metrics()), forced=True)
        if self.args.plot_metrics:
            self.dialog.plot_metrics()

        return success / n

def get_agent_type(model, smart=False):
    if isinstance(model, (RnnReferenceModel)):
        if smart:
            assert False
        else:
            return RnnAgent
    else:
        assert False, 'unknown model type: %s' % (model)
        

def main():
    parser = argparse.ArgumentParser(description='selfplaying script')
    parser.add_argument('--alice_model_file', type=str,
        help='Alice model file')
    parser.add_argument('--alice_forward_model_file', type=str,
        help='Alice forward model file')
    parser.add_argument('--bob_model_file', type=str,
        help='Bob model file')
    parser.add_argument('--context_file', type=str,
        help='context file')
    parser.add_argument('--temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--pred_temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--log_attention', action='store_true', default=False,
        help='log attention')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='print out converations')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--simple_posterior', action='store_true',
        help='use simple simple_posterior')
    parser.add_argument('--dec_use_attn', action='store_true',
        help='use attention for the decoder')
    parser.add_argument('--score_threshold', type=int, default=6,
        help='successful dialog should have more than score_threshold in score')
    parser.add_argument('--max_turns', type=int, default=20,
        help='maximum number of turns in a dialog')
    parser.add_argument('--log_file', type=str, default='selfplay.log',
        help='log dialogs to file')
    parser.add_argument('--smart_alice', action='store_true', default=False,
        help='make Alice smart again')
    parser.add_argument('--diverse_alice', action='store_true', default=False,
        help='make Alice smart again')
    parser.add_argument('--rollout_bsz', type=int, default=3,
        help='rollout batch size')
    parser.add_argument('--rollout_count_threshold', type=int, default=3,
        help='rollout count threshold')
    parser.add_argument('--smart_bob', action='store_true', default=False,
        help='make Bob smart again')
    parser.add_argument('--selection_model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--rollout_model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--diverse_bob', action='store_true', default=False,
        help='make Alice smart again')
    parser.add_argument('--ref_text', type=str,
        help='file with the reference text')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--domain', type=str, default='one_common',
        help='domain for the dialogue')
    parser.add_argument('--visual', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--eps', type=float, default=0.0,
        help='eps greedy')
    parser.add_argument('--data', type=str, default='data/onecommon',
        help='location of the data corpus')
    parser.add_argument('--unk_threshold', type=int, default=10,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--validate', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--plot_metrics', action='store_true', default=False,
        help='plot metrics')
    parser.add_argument('--markable_detector_file', type=str, default="markable_detector",
        help='visualize referents')
    parser.add_argument('--record_markables', action='store_true', default=False,
        help='record markables and referents')
    parser.add_argument('--repeat_selfplay', action='store_true', default=False,
        help='repeat selfplay')

    args = parser.parse_args()

    if args.repeat_selfplay:
        seeds = list(range(10))
    else:
        seeds = [args.seed]

    repeat_results = []

    for seed in seeds:
        utils.use_cuda(args.cuda)
        utils.set_seed(args.seed)

        if args.record_markables:
            if not os.path.exists(args.markable_detector_file + '_' + str(seed) + '.th'):
                assert False
            markable_detector = utils.load_model(args.markable_detector_file + '_' + str(seed) + '.th')
            if args.cuda:
                markable_detector.cuda()
            else:
                device = torch.device("cpu")
                markable_detector.to(device)
            markable_detector.eval()
            markable_detector_corpus = markable_detector.corpus_ty(domain, args.data, train='train_markable_{}.txt'.format(seed), valid='valid_markable_{}.txt'.format(seed), test='test_markable_{}.txt'.format(seed), #test='selfplay_reference_{}.txt'.format(seed),
                freq_cutoff=args.unk_threshold, verbose=True)
        else:
            markable_detector = None
            markable_detector_corpus = None

        alice_model = utils.load_model(args.alice_model_file + '_' + str(seed) + '.th')
        alice_ty = get_agent_type(alice_model, args.smart_alice)
        alice = alice_ty(alice_model, args, name='Alice', train=False, diverse=args.diverse_alice)
        alice.vis = args.visual

        bob_model = utils.load_model(args.bob_model_file + '_' + str(seed) + '.th')
        bob_ty = get_agent_type(bob_model, args.smart_bob)
        bob = bob_ty(bob_model, args, name='Bob', train=False, diverse=args.diverse_bob)

        bob.vis = False

        dialog = Dialog([alice, bob], args, markable_detector, markable_detector_corpus)
        ctx_gen = ContextGenerator(os.path.join(args.data, args.context_file + '.txt'))
        with open(os.path.join(args.data, args.context_file + '.json'), "r") as f:
            scenario_list = json.load(f)
        scenarios = {scenario['uuid']: scenario for scenario in scenario_list}
        logger = DialogLogger(verbose=args.verbose, log_file=args.log_file, scenarios=scenarios)

        selfplay = SelfPlay(dialog, ctx_gen, args, logger)
        result = selfplay.run()
        repeat_results.append(result)


    print("dump selfplay_markables.json")
    dump_json(dialog.selfplay_markables, "selfplay_markables.json")
    print("dump selfplay_referents.json")
    dump_json(dialog.selfplay_referents, "selfplay_referents.json")

    print("repeat selfplay results %.8f ( %.8f )" % (np.mean(repeat_results), np.std(repeat_results)))



if __name__ == '__main__':
    main()