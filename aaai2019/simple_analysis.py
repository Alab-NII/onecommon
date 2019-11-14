import argparse
from collections import defaultdict, Counter
from datetime import datetime
import json
import math
import os
import sys
import pdb
import re

from nltk import word_tokenize, bigrams

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.15)

min_shared = 4
max_shared = 6


def rgb_to_int(color):
    return int(re.search(r"[\d]+", color).group(0))


def get_id_to_shared(scenario_list):
    id_to_shared = {}
    for scenario in scenario_list:
        id_to_shared[scenario['uuid']] = scenario['shared']
    return id_to_shared


def dump_basic_statistics(chat_data, id_to_shared):
    results = defaultdict(dict)
    for case in range(min_shared, max_shared + 1):
        results[case]['total'] = 0
        results[case]['success'] = 0
        results[case]['total_time'] = 0
        results[case]['total_turns'] = 0
        results[case]['total_words'] = 0

    vocab = Counter()
    used_scenarios = set()

    for chat in chat_data:
        chat_id = chat['uuid']
        scenario_id = chat['scenario_uuid']
        scenario = chat['scenario']
        outcome = chat['outcome']
        events = chat['events']

        used_scenarios.add(scenario_id)
        case = id_to_shared[scenario_id]

        if len(events) > 0:
            results[case]['total'] += 1
            results[case]['success'] += int(outcome['reward'])
            for event in events:
                if event['action'] == "message":
                    msg = event['data']
                    results[case]['total_words'] += len(word_tokenize(msg))
                    vocab.update([w.lower() for w in word_tokenize(msg)])
                    results[case]['total_turns'] += 1

    total_dialogues = 0
    for case in range(min_shared, max_shared + 1):
        print("-- case {} --".format(case))
        print("total dialogs: {}".format(results[case]['total']))
        total_dialogues += results[case]['total']
        if results[case]['total'] <= 0:
            continue
        print("success rate: {:.5f}".format(results[case]['success'] / results[case]['total']))
        print("average tokens: {:.5f}".format(results[case]['total_words'] / results[case]['total']))
        print("average turns: {:.5f}".format(results[case]['total_turns'] / results[case]['total']))
        print("average tokens per turn: {:.5f}".format(results[case]['total_words'] / results[case]['total_turns']))
        print()
    print("total dialogues: {}".format(total_dialogues))
    print("vocabulary size: {}".format(len(vocab)))
    print("occupancy of top 10% frequent tokens: {:.5f}".format(sum([freq for w, freq in vocab.most_common(int(len(vocab) * 0.1))])
                                                             / sum(vocab.values())))


def plot_selection(chat_data, id_to_shared):
    min_color = 53
    max_color = 203
    min_size = 7
    max_size = 13
    color_bin = 5
    color_range = 1 + int((max_color - min_color) / color_bin)
    size_range = max_size - min_size + 1

    total = np.zeros((color_range, size_range))
    selected = np.zeros((color_range, size_range))

    def _group_color(color):
        return int((rgb_to_int(color) - min_color) / color_bin)

    def _group_size(size):
        return size - min_size

    for chat in chat_data:
        select_id = {}
        for kb in chat['scenario']['kbs']:
            for obj in kb:
                size = _group_size(obj['size'])
                color = _group_color(obj['color'])
                total[color][size] += 1
        for chat_event in chat['events']:
            if chat_event['action'] == 'select':
                select_id[chat_event['agent']] = chat_event['data']
        for agent in [0, 1]:
            for obj in chat['scenario']['kbs'][agent]:
                if obj['id'] == select_id[agent]:
                    size = _group_size(obj['size'])
                    color = _group_color(obj['color'])
                    selected[color][size] += 1
                    break

    ax = sns.heatmap((selected / total), cmap=cm.Blues, yticklabels=3)
    plt.xlabel('size', fontsize=18)
    plt.ylabel('color', fontsize=18)
    xticklabels = [str(int(x.get_text()) + min_size) for x in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels)
    yticklabels = [str(int(y.get_text())*color_bin + min_color) for y in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels)
    plt.tight_layout()
    plt.show()


def count_dict(chat_data, id_to_shared, word_dict):
    results = defaultdict(dict)
    for case in range(min_shared, max_shared + 1):
        results[case]['total'] = 0
        results[case]['total_turns'] = 0
        results[case]['total_words'] = 0

    vocab = Counter()
    for chat in chat_data:
        chat_id = chat['uuid']
        scenario_id = chat['scenario_uuid']
        scenario = chat['scenario']
        outcome = chat['outcome']
        events = chat['events']

        case = id_to_shared[scenario_id]

        if len(events) > 0:
            results[case]['total'] += 1
            for event in events:
                if event['action'] == "message":
                    msg = event['data']
                    results[case]['total_words'] += len(msg.split(" "))
                    uni = [w.lower() for w in word_tokenize(msg)]
                    bi = list((bigrams(uni)))
                    vocab.update(uni)
                    vocab.update(bi)
                    results[case]['total_turns'] += 1

    total_turns = 0
    total_tokens = 0
    for case in range(min_shared, max_shared + 1):
        total_turns += results[case]['total_turns']
        total_tokens += results[case]['total_words']

    nuance_dict = {}
    for nuance_type in word_dict.keys():
        count_type = 0
        type_dict = Counter()
        for key_word in word_dict[nuance_type]:
            if type(key_word) == dict:
                # key word is a bigram
                count_type += vocab[tuple([key_word['0'], key_word['1']])]
                type_dict[tuple([key_word['0'], key_word['1']])] = vocab[tuple([key_word['0'], key_word['1']])]
            else:
                # key word is a unigram
                count_type += vocab[key_word]
                type_dict[key_word] = vocab[key_word]
        print("{}: {} ({:.5f} per 100 utterances)".format(nuance_type, count_type, 100.0 * count_type / total_turns))
        nuance_dict[nuance_type] = type_dict


def word_cloud(chat_data):
    from wordcloud import WordCloud

    tokens = []
    text = ""

    for chat in chat_data:
        chat_id = chat['uuid']
        scenario_id = chat['scenario_uuid']
        scenario = chat['scenario']
        outcome = chat['outcome']
        events = chat['events']

        for event in events:
            if event['action'] == "message":
                msg = event['data']
                tokens += [w.lower() for w in word_tokenize(msg)]
                text += msg

    wordcloud = WordCloud(max_font_size=64, max_words=160,
                          width=1280, height=640,
                          background_color="black").generate(' '.join(tokens))
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default="data")
    parser.add_argument('--scenario_file', type=str,
                        default="aaai_train_scenarios.json")
    parser.add_argument('--scenario_file_2', type=str,
                        default="aaai_train_scenarios_2.json")
    parser.add_argument('--transcript_file', type=str,
                        default="final_transcripts.json")
    parser.add_argument('--nuance_dict', type=str,
                        default="nuance_dict.json")
    # analyses to conduct
    parser.add_argument('--basic_statistics', action='store_true', default=False)
    parser.add_argument('--count_dict', action='store_true', default=False)
    parser.add_argument('--plot_selection_bias', action='store_true', default=False)
    parser.add_argument('--word_cloud', action='store_true', default=False)

    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise ValueError("data directory does not exist")
    with open(os.path.join(args.data, args.transcript_file), "r") as f:
        chat_data = json.load(f)
    with open(os.path.join(args.data, args.scenario_file), "r") as f:
        scenario_list = json.load(f)
    with open(os.path.join(args.data, args.scenario_file_2), "r") as f:
        scenario_list += json.load(f)
    with open(os.path.join(args.data, args.nuance_dict), "r") as f:
        nuance_dict = json.load(f)

    id_to_shared = get_id_to_shared(scenario_list)

    if args.basic_statistics:
        dump_basic_statistics(chat_data, id_to_shared)
    if args.count_dict:
        count_dict(chat_data, id_to_shared, nuance_dict)
    if args.plot_selection_bias:
        plot_selection(chat_data, id_to_shared)
    if args.word_cloud:
        word_cloud(chat_data)
