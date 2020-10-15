"""
Performs evaluation of the model on the test dataset.
"""

import argparse
import copy
import json
import os
import pdb
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import data
import utils
from engines import Criterion
from domain import get_domain

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import pandas as pd
sns.set(font_scale=1.15)

def read_json(path):
    try:
        return json.load(open(path))
    except:
        raise Exception('Error reading JSON from %s' % path)

def dump_json(file, path):
    try:
        with open(path, "w") as fout:
            json.dump(file, fout, indent=4, sort_keys=True)
    except:
        raise Exception('Error writing JSON to %s' % path)

def reference_to_svg(kb, ref_out):
    svg = '''<svg id="svg" width="430" height="430"><circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>\n'''
    for i, obj in enumerate(kb):
        svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"{3}\" />\n".format(obj['x'], obj['y'],
                                                                                             obj['size'], obj['color'])
        if ref_out[i] == 1:
            svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"none\" stroke=\"{3}\" stroke-width=\"4\" stroke-dasharray=\"3,3\"\n/>".format(obj['x'], obj['y'],
                        obj['size'] + 4, "green")
    svg += '''</svg>'''
    return svg


def main():
    parser = argparse.ArgumentParser(description='testing script for reference resolution')
    parser.add_argument('--data', type=str, default='data/onecommon',
        help='location of the data corpus')
    parser.add_argument('--unk_threshold', type=int, default=10,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--model_file', type=str, required=True,
        help='pretrained model file')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--hierarchical', action='store_true', default=False,
        help='use hierarchical model')
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--domain', type=str, default='one_common',
        help='domain for the dialogue')
    parser.add_argument('--vocab_corpus', choices=['full', 'uncorrelated', 'success_only'], default='full',
        help='vocabulary of the corpus to use')
    parser.add_argument('--corpus_type', choices=['full', 'uncorrelated', 'success_only'], default='full',
        help='type of test corpus to use')
    parser.add_argument('--bleu_n', type=int, default=0,
        help='test ngram bleu')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='temperature')

    # for error analysis
    parser.add_argument('--transcript_file', type=str, default='final_transcripts.json',
        help='scenario file')
    parser.add_argument('--markable_file', type=str, default='markable_annotation.json',
        help='scenario file')
    parser.add_argument('--show_errors', action='store_true', default=False,
        help='show errors')

    # analysis parameters
    parser.add_argument('--fix_misspellings', action='store_true', default=False,
        help='fix misspellings')
    parser.add_argument('--shuffle_utterance', action='store_true', default=False,
        help='shuffle order of words in the utterance')
    parser.add_argument('--shuffle_word_types', type=str, nargs='*', default=[],
        help='shuffle specified class of words in the output')
    parser.add_argument('--drop_word_types', type=str, nargs='*', default=[],
        help='drop specified class of words in the output')
    parser.add_argument('--replace_word_types', type=str, nargs='*', default=[],
        help='replace specified class of words in the output')
    parser.add_argument('--repeat_test', action='store_true', default=False,
        help='repeat training n times')
    parser.add_argument('--test_ref_forward', action='store_true', default=False,
        help='test forward reference instead')

    args = parser.parse_args()

    if args.bleu_n > 0:
        # current support
        args.bsz = 1

    if args.repeat_test:
        seeds = list(range(10))
    else:
        seeds = [args.seed]

    repeat_results = defaultdict(list)

    model_referent_annotation = {}

    init2num_referents = defaultdict(Counter)

    for seed in seeds:
        device_id = utils.use_cuda(args.cuda)
        utils.set_seed(args.seed)

        domain = get_domain(args.domain)
        model = utils.load_model(args.model_file + '_' + str(seed) + '.th')
        if args.cuda:
            model.cuda()
        else:
            device = torch.device("cpu")
            model.to(device)
        model.eval()

        corpus = model.corpus_ty(domain, args.data, train='train_reference_shift_{}.txt'.format(seed), valid='valid_reference_shift_{}.txt'.format(seed), test='test_reference_shift_{}.txt'.format(seed),
            freq_cutoff=args.unk_threshold, verbose=True)

        with open(os.path.join(args.data, args.transcript_file), "r") as f:
            dialog_corpus = json.load(f)
        with open(os.path.join(args.data, args.markable_file), "r") as f:
            markable_annotation = json.load(f)
        with open(os.path.join(args.data, "aggregated_referent_annotation.json"), "r") as f:
            aggregated_referent_annotation = json.load(f)

        scenarios = {scenario['scenario_uuid']: scenario for scenario in dialog_corpus}

        crit = Criterion(model.word_dict, device_id=device_id)
        sel_crit = nn.CrossEntropyLoss()
        ref_crit = nn.BCEWithLogitsLoss()

        testset, testset_stats = corpus.test_dataset(args.bsz)
        test_lang_loss, test_select_loss, test_reference_loss, test_select_correct, test_select_total, test_reference_correct, test_reference_total, test_num_ref_correct, test_num_ref_total  = 0, 0, 0, 0, 0, 0, 0, 0, 0

        """
            Variables to keep track of the results for analysis
        """

        # num_referents --> count, count correct
        num_markables = 0
        num_markables_counter = Counter()
        num_markables_correct = Counter()

        exact_match = 0
        exact_match_counter = Counter()

        # location of markable --> count, count correct, count exact match
        location_counter = Counter()
        location_correct = Counter()
        location_exact_match = Counter()

        # information to compute correlation between selection and reference score 
        select_correct = {}
        reference_correct = {}
        reference_total = {}

        # markable text --> count, count correct, count exact match
        text_counter = Counter()
        text_correct = Counter()
        text_exact_match = Counter()

        # init token --> count, count correct
        init_counter = Counter()
        init_correct = Counter()
        init_exact_match = Counter()

        # num ref confusion
        num_ref_confusion = np.zeros([8, 8], dtype=int)

        anaphora_list = ["it", "that", "thats", "this", "its", "they", "their", "itself", "them", "those", "it's"]
        total_anaphora = 0
        correct_anaphora = 0

        bleu_scores = []

        for batch in testset:
            ctx, inpt, tgt, ref_inpt, ref_tgt, num_ref_tgt, sel_tgt, scenario_ids, real_ids, agents, chat_ids, sel_idx = batch

            ctx = Variable(ctx)
            inpt = Variable(inpt)
            if ref_inpt is not None:
                ref_inpt = Variable(ref_inpt)
            out, ref_out, num_ref_out, sel_out = model.forward(ctx, inpt, ref_inpt, sel_idx)

            tgt = Variable(tgt)
            sel_tgt = Variable(sel_tgt)
            lang_loss = crit(out, tgt)

            if ref_inpt is not None:
                ref_tgt = Variable(ref_tgt)
                ref_tgt = torch.transpose(ref_tgt, 0, 1).contiguous().float()
                ref_loss = ref_crit(ref_out, ref_tgt)
                t = Variable(torch.FloatTensor([0])) # threshold
                if model.args.num_ref_weight > 0:
                    num_ref_pred = num_ref_out.max(dim=2)[1]
                    ref_results = torch.zeros_like(ref_tgt)
                    ref_correct = 0
                    for i in range(ref_out.size(0)):
                        for j in range(ref_out.size(1)):
                            ref_pred = torch.zeros_like(ref_tgt[i][j])
                            for ref_idx in range(ref_pred.size(0)):
                                if ref_idx in ref_out[i][j].topk(num_ref_pred[i][j])[1]:
                                    ref_pred[ref_idx] = 1.0
                            ref_results[i][j] = (ref_pred.long() == ref_tgt[i][j].long())
                            ref_correct += (ref_pred.long() == ref_tgt[i][j].long()).sum().item()
                    ref_total = ref_tgt.size(0) * ref_tgt.size(1) * ref_tgt.size(2)
                else:
                    ref_results = ((ref_out > 0).long() == ref_tgt.long())
                    ref_correct = ((ref_out > 0).long() == ref_tgt.long()).sum().item()
                    ref_total = ref_tgt.size(0) * ref_tgt.size(1) * ref_tgt.size(2)

                # compute more details of reference resolution
                for i in range(ref_tgt.size(0)): # markable idx
                    for j in range(ref_tgt.size(1)): # batch idx
                        chat_id = chat_ids[j]

                        # add chat level details if not exists
                        if chat_id not in reference_correct:
                            reference_correct[chat_id] = ref_results[:,j,:].sum().item()
                        if chat_id not in reference_total:
                            reference_total[chat_id] = ref_results[:,j,:].size(0) * ref_results[:,j,:].size(1)
                        if chat_id not in model_referent_annotation:
                            model_referent_annotation[chat_id] = {}

                        markables = []
                        # markables information from aggregated_referent_annotation
                        for markable in markable_annotation[chat_id]["markables"]:
                            markable_id = markable["markable_id"]
                            if markable_id in aggregated_referent_annotation[chat_id] and markable["speaker"] == agents[j]:
                                if "unidentifiable" in aggregated_referent_annotation[chat_id][markable_id] and aggregated_referent_annotation[chat_id][markable_id]["unidentifiable"]:
                                    if markable_id not in model_referent_annotation[chat_id] and markable["speaker"] == agents[j]:
                                        model_referent_annotation[chat_id][markable_id] = {'ambiguous': False, 'referents': [], 'unidentifiable': True}
                                    continue
                                markables.append(markable)
                        assert len(markables) == ref_tgt.size(0)

                        if model.args.num_ref_weight > 0:
                            ref_pred = torch.zeros_like(ref_tgt[i][j])
                            for ref_idx in range(ref_pred.size(0)):
                                #if ref_idx in ref_out[i][j].topk(num_ref_tgt[i][j])[1]:
                                if ref_idx in ref_out[i][j].topk(num_ref_pred[i][j])[1]:
                                    ref_pred[ref_idx] = 1.0
                            correct_result = (ref_pred.long() == ref_tgt[i][j].long()).sum().item()
                            exact_match_result = torch.equal(ref_pred.long(), ref_tgt[i][j].long())
                            num_referents = ref_tgt[i][j].long().sum().item()
                        else:
                            correct_result = ((ref_out > 0).long()[i][j] == ref_tgt.long())[i][j].sum().item()
                            exact_match_result = torch.equal((ref_out > 0).long()[i][j], ref_tgt.long()[i][j])
                            num_referents = ref_tgt.long()[i][j].sum().item()
                            ref_pred = (ref_out > 0).long()[i][j]

                        """
                            Add information to variables
                        """
                        num_markables += 1
                        num_markables_counter[num_referents] += 1
                        num_markables_correct[num_referents] += correct_result

                        # compute exact match 
                        if exact_match_result:
                            exact_match += 1
                            exact_match_counter[ref_tgt.long()[i][j].sum().item()] += 1
                            location_exact_match[i] += 1
                            if num_referents == 1: # temporal condition
                                text_exact_match[markables[i]["text"].lower()] += 1
                                init_exact_match[markables[i]["text"].lower().split(" ")[0]] += 1

                        location_correct[i] += correct_result
                        location_counter[i] += 1

                        if num_referents == 1: # temporal condition
                            text_counter[markables[i]["text"].lower()] += 1
                            text_correct[markables[i]["text"].lower()] += correct_result
                            init_counter[markables[i]["text"].lower().split(" ")[0]] += 1
                            init_correct[markables[i]["text"].lower().split(" ")[0]] += correct_result

                        init2num_referents[markables[i]["text"].lower().split(" ")[0]][num_referents] += 1

                        # test anaphora
                        if markables[i]["text"].lower() in anaphora_list:
                            total_anaphora += 1
                            if exact_match_result:
                                correct_anaphora += 1

                        # keep track of model predictions for later visualization
                        chat = [chat for chat in dialog_corpus if chat['uuid'] == chat_id]
                        chat = chat[0]
                        if markables[i]['markable_id'] not in model_referent_annotation[chat_id]:
                            model_referent_annotation[chat_id][markables[i]['markable_id']] = {}
                            model_referent_annotation[chat_id][markables[i]['markable_id']]['referents'] = []
                            model_referent_annotation[chat_id][markables[i]['markable_id']]['ambiguous'] = False
                            model_referent_annotation[chat_id][markables[i]['markable_id']]['unidentifiable'] = False
                            for ent, is_referent in zip(chat['scenario']['kbs'][agents[j]], ref_pred.long().tolist()):
                            #for ent, is_referent in zip(chat['scenario']['kbs'][agents[j]], (ref_out > 0).long()[i][j].tolist()):
                                if is_referent:
                                    model_referent_annotation[chat_id][markables[i]['markable_id']]['referents'].append("agent_{}_{}".format(agents[j], ent['id']))
            else:
                ref_loss = None
                ref_correct = 0
                ref_total = 0

            sel_loss = sel_crit(sel_out, sel_tgt)
            sel_correct = (sel_out.max(dim=1)[1] == sel_tgt).sum().item()
            sel_total = sel_out.size(0)
            for i in range(sel_tgt.size(0)): # batch idx
                chat_id = chat_ids[i]
                sel_resuts = (sel_out.max(dim=1)[1] == sel_tgt)
                if sel_resuts[i]:
                    select_correct[chat_id] = 1
                else:
                    select_correct[chat_id] = 0

            if model.args.num_ref_weight > 0 and num_ref_out is not None:
                num_ref_out = num_ref_out.view(-1, num_ref_out.size(2))
                num_ref_tgt = torch.transpose(num_ref_tgt, 0, 1).contiguous()
                num_ref_tgt = num_ref_tgt.view(-1)
                num_ref_loss = sel_crit(num_ref_out, num_ref_tgt)
                num_ref_correct = (num_ref_out.max(dim=1)[1] == num_ref_tgt).sum().item()
                num_ref_total = num_ref_tgt.size(0)
                for mi in range(num_ref_out.size(0)):
                    model_pred = num_ref_out[mi].max(dim=0)[1].item()
                    ground_truth = num_ref_tgt[mi].item()
                    num_ref_confusion[ground_truth][model_pred] += 1
            else:
                num_ref_correct = 0
                num_ref_total = 0

            test_lang_loss += lang_loss.item()
            test_select_loss += sel_loss.item()
            if ref_loss:
                test_reference_loss += ref_loss.item()
            test_select_correct += sel_correct
            test_select_total += sel_total
            test_reference_correct += ref_correct
            test_reference_total += ref_total
            test_num_ref_correct += num_ref_correct
            test_num_ref_total += num_ref_total           

            if args.bleu_n > 0:
                ctx_h = model.ctx_encoder(ctx.transpose(0,1))

                my_utterance = None
                idx = 0
                while True:
                    if inpt[idx] == model.word_dict.word2idx['YOU:']:
                        start = idx
                        my_utterance = model.read_and_write(
                            inpt[:idx], ctx_h, 30, temperature=args.temperature)
                        my_utterance = model.word_dict.i2w(my_utterance)
                        #print(my_utterance)
                        while not inpt[idx] in [model.word_dict.word2idx[stop_token] for stop_token in data.STOP_TOKENS]:
                            idx += 1
                        end = idx
                        golden_utterance = inpt[start:end]
                        golden_utterance = model.word_dict.i2w(golden_utterance)
                        bleu_scores.append(100 * sentence_bleu([golden_utterance], my_utterance, weights=[1 for i in range(4) if args.bleu_n == i], #weights=[1 / args.bleu_n] * args.bleu_n,
                                                               smoothing_function=SmoothingFunction().method7))
                    if inpt[idx] == model.word_dict.word2idx['<selection>']:
                        break

                    idx += 1

        # Main results:
        # Dividing by the number of words in the input, not the tokens modeled,
        # because the latter includes padding
        test_lang_loss /= testset_stats['nonpadn']
        test_select_loss /= len(testset)
        test_select_accuracy = test_select_correct / test_select_total
        test_reference_accuracy = test_reference_correct / test_reference_total
        if test_num_ref_total > 0:
            test_num_ref_accuracy = test_num_ref_correct / test_num_ref_total
        else:
            test_num_ref_accuracy = 0
        print('testlangloss %.8f | testlangppl %.8f' % (test_lang_loss, np.exp(test_lang_loss)))
        print('testselectloss %.8f | testselectaccuracy %.6f' % (test_select_loss, test_select_accuracy))
        print('testreferenceloss %.8f | testreferenceaccuracy %.6f' % (test_reference_loss, test_reference_accuracy))
        print('reference_exact_match %.6f' % (exact_match / num_markables))
        for k in num_markables_counter.keys():
            print('{}: {:.4f} {:.4f} (out of {})'.format(k, num_markables_correct[k] / (num_markables_counter[k] * 7), exact_match_counter[k] / num_markables_counter[k], num_markables_counter[k]))
        print('test anaphora: {} (out of {})'.format(correct_anaphora / total_anaphora, total_anaphora))

        if args.bleu_n > 0:
            print('average bleu score {}'.format(np.mean(bleu_scores)))

        # reference/selection correlation
        reference_score = []
        selection_score = []
        for chat_id in reference_correct.keys():
            reference_score.append(reference_correct[chat_id] / reference_total[chat_id])
            selection_score.append(select_correct[chat_id])
        plt.xlabel('reference score', fontsize=14)
        plt.ylabel('selection score', fontsize=14)
        sns.regplot(x=reference_score, y=selection_score)
        plt.savefig('reference_selection_{}.png'.format(seed), dpi=300)
        plt.clf()
        reference_score = np.array(reference_score)
        selection_score = np.array(selection_score)
        print("reference selection correlation: {}".format(np.corrcoef(reference_score, selection_score)))

        # keep track of results for this run
        repeat_results["test_lang_loss"].append(test_lang_loss)
        repeat_results["test_select_loss"].append(test_select_loss)
        repeat_results["test_select_accuracy"].append(test_select_accuracy)
        repeat_results["test_reference_loss"].append(test_reference_loss)
        repeat_results["test_reference_accuracy"].append(test_reference_accuracy)
        repeat_results["test_num_ref_accuracy"].append(test_num_ref_accuracy)
        repeat_results["correlation_score"].append(np.corrcoef(reference_score, selection_score)[0][1])
        repeat_results["num_markables_counter"].append(copy.copy(num_markables_counter))
        repeat_results["exact_match_counter"].append(copy.copy(exact_match_counter))
        repeat_results["num_markables_correct"].append(copy.copy(num_markables_correct))
        repeat_results["reference_exact_match"].append(exact_match / num_markables)
        repeat_results["test_perplexity"].append(np.exp(test_lang_loss))
        repeat_results["location_counter"].append(copy.copy(location_counter))
        repeat_results["location_correct"].append(copy.copy(location_correct))
        repeat_results["location_exact_match"].append(copy.copy(location_exact_match))
        repeat_results["init_counter"].append(copy.copy(init_counter))
        repeat_results["init_correct"].append(copy.copy(init_correct))
        repeat_results["init_exact_match"].append(copy.copy(init_exact_match))

    print("=================================\n\n")
    print("repeat test lang loss %.8f" % np.mean(repeat_results["test_lang_loss"]))
    print("repeat test select loss %.8f" % np.mean(repeat_results["test_select_loss"]))
    print("repeat test select accuracy %.8f ( %.8f )" % (np.mean(repeat_results["test_select_accuracy"]), np.std(repeat_results["test_select_accuracy"])))
    print("repeat test reference loss %.8f" % np.mean(repeat_results["test_reference_loss"]))
    print("repeat test reference accuracy %.8f ( %.8f )" % (np.mean(repeat_results["test_reference_accuracy"]), np.std(repeat_results["test_reference_accuracy"])))
    print("repeat test num ref accuracy %.8f ( %.8f )" % (np.mean(repeat_results["test_num_ref_accuracy"]), np.std(repeat_results["test_reference_accuracy"])))
    print("repeat correlation score %.8f ( %.8f )" % (np.mean(repeat_results["correlation_score"]), np.std(repeat_results["correlation_score"])))
    print("repeat correlation score %.8f ( %.8f )" % (np.mean(repeat_results["correlation_score"]), np.std(repeat_results["correlation_score"])))
    print("repeat reference exact match %.8f ( %.8f )" % (np.mean(repeat_results["reference_exact_match"]), np.std(repeat_results["reference_exact_match"])))
    print("repeat test perplexity %.8f ( %.8f )" % (np.mean(repeat_results["test_perplexity"]), np.std(repeat_results["test_perplexity"])))

    for k in num_markables_counter.keys():
        print("repeat accuracy and exact match:")
        num_markables = []
        exact_match = []
        exact_match_rate = []
        num_markables_correct = []
        for seed in range(len(seeds)):
            num_markables.append(repeat_results["num_markables_counter"][seed][k])
            exact_match.append(repeat_results["exact_match_counter"][seed][k])
            exact_match_rate.append(repeat_results["exact_match_counter"][seed][k] / repeat_results["num_markables_counter"][seed][k])
            num_markables_correct.append(repeat_results["num_markables_correct"][seed][k] / (repeat_results["num_markables_counter"][seed][k] * 7))
        print('{}: {:.5f} (std {}) {:.5f} (std {}) (count {})'.format(k, np.mean(num_markables_correct), np.std(num_markables_correct), np.mean(exact_match_rate), np.std(exact_match_rate), np.mean(num_markables)))

    dump_json(model_referent_annotation, "{}_referent_annotation.json".format(args.model_file))

    print("exact match at each location:")
    markable_location_plot = []
    exact_match_rate_plot = []
    accuracy_plot = []
    for loc in range(12):
        accuracy = []
        exact_match_rate = []
        total_count = 0
        for seed in range(len(seeds)):
            if repeat_results["location_counter"][seed][loc] > 0:
                exact_match_rate.append(repeat_results["location_exact_match"][seed][loc] / repeat_results["location_counter"][seed][loc])
                total_count += repeat_results["location_counter"][seed][loc]
                markable_location_plot.append(loc + 1)
                exact_match_rate_plot.append(repeat_results["location_exact_match"][seed][loc] / repeat_results["location_counter"][seed][loc])
                accuracy_plot.append(repeat_results["location_correct"][seed][loc] / (7 * repeat_results["location_counter"][seed][loc]))
        if len(exact_match_rate) > 0:
            print('Loc @ {}: {:.5f} (std {:.5f}) (valid runs: {}, total: {})'.format(loc + 1, np.mean(exact_match_rate), np.std(exact_match_rate), len(exact_match_rate), total_count))

    plt.xlabel('markable location', fontsize=14)
    plt.ylabel('exact match rate', fontsize=14)
    sns.lineplot(x=markable_location_plot, y=exact_match_rate_plot)
    plt.savefig('location_exact_match_rate.png', dpi=300)
    plt.clf()

    plt.xlabel('markable location', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    sns.lineplot(x=markable_location_plot, y=accuracy_plot)
    plt.savefig('location_accuracy.png', dpi=300)
    plt.clf()

    plt.xlabel('markable position', fontsize=14)
    plt.ylabel('percentage', fontsize=14)
    sns.lineplot(x=markable_location_plot, y=accuracy_plot, legend="brief", label="accuracy")
    sns.lineplot(x=markable_location_plot, y=exact_match_rate_plot, legend="brief", label="exact match")
    plt.savefig('location_results.png', dpi=300)
    plt.clf()


    print("compute results based on initial token:")
    #for tok in model.word_dict.w2i.keys():
    definite_toks = ["the"]
    indefinite_toks = ["a", "an"]
    definite_accuracies = []
    indefinite_accuracies = []
    other_accuracies = []
    definite_exact_matches = []
    indefinite_exact_matches = []
    other_exact_matches = []
    definite_counts = []
    indefinite_counts = []
    other_counts = []
    for seed in range(len(seeds)):
        num_correct = 0
        num_exact_match = 0
        num_total = 0
        for tok in definite_toks:
            num_total += repeat_results["init_counter"][seed][tok]
            num_correct += repeat_results["init_correct"][seed][tok]
            num_exact_match += repeat_results["init_exact_match"][seed][tok]
        definite_accuracies.append(num_correct / (7 * num_total))
        definite_exact_matches.append(num_exact_match / num_total)
        definite_counts.append(num_total)

        num_correct = 0
        num_exact_match = 0
        num_total = 0
        for tok in indefinite_toks:
            num_total += repeat_results["init_counter"][seed][tok]
            num_correct += repeat_results["init_correct"][seed][tok]
            num_exact_match += repeat_results["init_exact_match"][seed][tok]
        indefinite_accuracies.append(num_correct / (7 * num_total))
        indefinite_exact_matches.append(num_exact_match / num_total)
        indefinite_counts.append(num_total)

        num_correct = 0
        num_exact_match = 0
        num_total = 0
        for tok in repeat_results["init_counter"][seed].keys():
            if tok not in definite_toks + indefinite_toks:
                num_total += repeat_results["init_counter"][seed][tok]
                num_correct += repeat_results["init_correct"][seed][tok]
                num_exact_match += repeat_results["init_exact_match"][seed][tok]
        other_accuracies.append(num_correct / (7 * num_total))
        other_exact_matches.append(num_exact_match / num_total)
        other_counts.append(num_total)

    print("definite: accuracies {} (std {}), exact match rate {} (std {}), total count {} (std {})".format(
        np.mean(definite_accuracies), np.std(definite_accuracies),
        np.mean(definite_exact_matches), np.std(definite_exact_matches),
        np.mean(definite_counts), np.std(definite_counts)))

    print("indefinite: accuracies {} (std {}), exact match rate {} (std {}), total count {} (std {})".format(
        np.mean(indefinite_accuracies), np.std(indefinite_accuracies),
        np.mean(indefinite_exact_matches), np.std(indefinite_exact_matches),
        np.mean(indefinite_counts), np.std(indefinite_counts)))

    print("other: accuracies {} (std {}), exact match rate {} (std {}), total count {} (std {})".format(
        np.mean(other_accuracies), np.std(other_accuracies),
        np.mean(other_exact_matches), np.std(other_exact_matches),
        np.mean(other_counts), np.std(other_counts)))

    valid_markables = 0
    for chat_id in model_referent_annotation.keys():
        for markable_id in model_referent_annotation[chat_id].keys():
            if 'unidentifiable' in aggregated_referent_annotation[chat_id][markable_id] and aggregated_referent_annotation[chat_id][markable_id]['unidentifiable']:
                continue
            valid_markables += 1
    print("model valid markables: {}".format(valid_markables))

    valid_markables = 0
    for chat_id in aggregated_referent_annotation.keys():
        for markable_id in aggregated_referent_annotation[chat_id].keys():
            if 'unidentifiable' in aggregated_referent_annotation[chat_id][markable_id] and aggregated_referent_annotation[chat_id][markable_id]['unidentifiable']:
                continue
            valid_markables += 1
    print("aggregated valid markables: {}".format(valid_markables))


if __name__ == '__main__':
    main()