import os
import random
import sys
import pdb
import copy
import re
from collections import OrderedDict, defaultdict

import torch
import numpy as np

# special tokens
SPECIAL = [
    '<eos>',
    '<unk>',
    '<selection>',
    '<pad>',
]

# tokens that stops either a sentence or a conversation
STOP_TOKENS = [
    '<eos>',
    '<selection>',
]


def get_tag(tokens, tag):
    """Extracts the value inside the given tag."""
    return tokens[tokens.index('<' + tag + '>') + 1:tokens.index('</' + tag + '>')]


def to_float(tokens):
    return [float(token) for token in tokens.split()]


def read_lines(file_name):
    """Reads all the lines from the file."""
    assert os.path.exists(file_name), 'file does not exists %s' % file_name
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


class Dictionary(object):
    """Maps words into indeces.

    It has forward and backward indexing.
    """

    def __init__(self, init=True):
        self.word2idx = OrderedDict()
        self.idx2word = []
        if init:
            # add special tokens if asked
            for i, k in enumerate(SPECIAL):
                self.word2idx[k] = i
                self.idx2word.append(k)

    def add_word(self, word):
        """Adds a new word, if the word is in the dictionary, just returns its index."""
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def i2w(self, idx):
        """Converts a list of indeces into words."""
        return [self.idx2word[i] for i in idx]

    def w2i(self, words):
        """Converts a list of words into indeces. Uses <unk> for the unknown words."""
        unk = self.word2idx.get('<unk>', None)
        return [self.word2idx.get(w, unk) for w in words]

    def get_idx(self, word):
        """Gets index for the word."""
        unk = self.word2idx.get('<unk>', None)
        return self.word2idx.get(word, unk)

    def get_word(self, idx):
        """Gets word by its index."""
        return self.idx2word[idx]

    def __len__(self):
        return len(self.idx2word)

    def read_tag(file_name, tag, freq_cutoff=-1, init_dict=True):
        """Extracts all the values inside the given tag.

        Applies frequency cuttoff if asked.
        """
        token_freqs = OrderedDict()
        with open(file_name, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                tokens = get_tag(tokens, tag)
                for token in tokens:
                    token_freqs[token] = token_freqs.get(token, 0) + 1
        dictionary = Dictionary(init=init_dict)
        token_freqs = sorted(token_freqs.items(),
                             key=lambda x: x[1], reverse=True)
        for token, freq in token_freqs:
            if freq > freq_cutoff:
                dictionary.add_word(token)
        return dictionary

    def from_file(file_name, freq_cutoff):
        """Constructs a dictionary from the given file."""
        assert os.path.exists(file_name)
        word_dict = Dictionary.read_tag(
            file_name, 'dialogue', freq_cutoff=freq_cutoff)
        return word_dict


class WordCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, domain, path, freq_cutoff=2, train='train.txt',
                 valid='valid.txt', test='test.txt', verbose=False, word_dict=None):
        self.verbose = verbose
        if word_dict is None:
            self.word_dict = Dictionary.from_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff)
        else:
            self.word_dict = word_dict

        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[1]) for x in self.train])

    def tokenize(self, file_name):
        """Tokenizes the file and produces a dataset."""
        lines = read_lines(file_name)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_vals = [float(val) for val in get_tag(tokens, 'input')]
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))
            output_idx = int(get_tag(tokens, 'output')[0])
            scenario_id = get_tag(tokens, 'scenario_id')[0]
            real_ids = get_tag(tokens, 'real_ids')
            agent = int(get_tag(tokens, 'agent')[0])
            dataset.append((input_vals, word_idxs, output_idx, scenario_id, real_ids, agent))
            # compute statistics
            total += len(word_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.copy(self.train), bsz, shuffle=shuffle, name="train")

    def valid_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.valid), bsz, shuffle=shuffle, name="valid")

    def test_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle, name="test")

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        """Splits given dataset into batches."""
        if shuffle:
            random.shuffle(dataset)

        # sort by dialog length and pad
        dataset.sort(key=lambda x: len(x[1]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0,
        }

        for i in range(0, len(dataset), bsz):
            inputs, words, output, scenario_ids, real_ids, agents = [], [], [], [], [], []
            for j in range(i, min(i + bsz, len(dataset))):
                inputs.append(dataset[j][0])
                words.append(dataset[j][1])
                output.append(dataset[j][2])
                scenario_ids.append(dataset[j][3])
                real_ids.append(dataset[j][4])
                agents.append(dataset[j][5])

            # the longest dialogue in the batch
            max_len = len(words[-1])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                # one additional pad
                words[j] += [pad] * (max_len - len(words[j]) + 1)

            # construct tensor for context
            ctx = torch.Tensor(inputs).float()
            data = torch.Tensor(words).long().transpose(0, 1).contiguous()
            # construct tensor for selection target
            sel_tgt = torch.Tensor(output).long()
            if device is not None:
                ctx = ctx.to(device)
                data = data.to(device)
                sel_tgt = sel_tgt.to(device)

            # construct tensor for input and target
            inpt = data.narrow(0, 0, data.size(0) - 1)
            tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

            batches.append((ctx, inpt, tgt, sel_tgt, scenario_ids, real_ids, agents))

        if shuffle:
            random.shuffle(batches)

        return batches, stats


class ReferenceCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, domain, path, freq_cutoff=2, train='train_reference.txt',
                 valid='valid_reference.txt', test='test_reference.txt', verbose=False, word_dict=None):
        self.verbose = verbose
        if word_dict is None:
            self.word_dict = Dictionary.from_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff)
        else:
            self.word_dict = word_dict

        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[1]) for x in self.train])

    def tokenize(self, file_name):
        """Tokenizes the file and produces a dataset."""
        lines = read_lines(file_name)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_vals = [float(val) for val in get_tag(tokens, 'input')]
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))
            referent_idxs = [int(val) for val in get_tag(tokens, 'referents')]
            output_idx = int(get_tag(tokens, 'output')[0])
            scenario_id = get_tag(tokens, 'scenario_id')[0]
            real_ids = get_tag(tokens, 'real_ids')
            agent = int(get_tag(tokens, 'agent')[0])
            chat_id = get_tag(tokens, 'chat_id')[0]
            dataset.append((input_vals, word_idxs, referent_idxs, output_idx, scenario_id, real_ids, agent, chat_id))
            # compute statistics
            total += len(word_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.copy(self.train), bsz, shuffle=shuffle, name="train")

    def valid_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.valid), bsz, shuffle=shuffle, name="valid")

    def test_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle, name="test")

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        """Splits given dataset into batches."""
        if shuffle:
            random.shuffle(dataset)

        # sort by markable length and pad
        dataset.sort(key=lambda x: len(x[2]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0,
        }

        i = 0
        while i < len(dataset):
            markable_length = len(dataset[i][2])

            ctxs, dials, refs, sels, scenario_ids, real_ids, agents, chat_ids, sel_idxs = [], [], [], [], [], [], [], [], []

            for _ in range(bsz):
                if i >= len(dataset) or len(dataset[i][2]) != markable_length:
                    break
                ctxs.append(dataset[i][0])
                dials.append(dataset[i][1])
                refs.append(dataset[i][2])
                sels.append(dataset[i][3])
                scenario_ids.append(dataset[i][4])
                real_ids.append(dataset[i][5])
                agents.append(dataset[i][6])
                chat_ids.append(dataset[i][7])
                sel_idxs.append(len(dataset[i][1]) - 1)
                i += 1

            # the longest dialogue in the batch
            max_len = max([len(dial) for dial in dials])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(dials)):
                stats['n'] += max_len
                stats['nonpadn'] += len(dials[j])
                # one additional pad
                dials[j] += [pad] * (max_len - len(dials[j]) + 1)

            # construct tensor for context (bsz, num_ent * dim_ent)
            ctx = torch.Tensor(ctxs).float()
            
            # dialog data (seq_len, bsz)
            data = torch.Tensor(dials).long().transpose(0, 1).contiguous()

            # construct tensor for reference target
            num_markables = int(markable_length / 10)

            ref_inpt = []
            ref_tgt = []
            for j in range(len(refs)):
                _ref_inpt = []
                _ref_tgt = []
                for k in range(num_markables):
                    _ref_inpt.append(refs[j][10 * k: 10 * k + 3])
                    _ref_tgt.append(refs[j][10 * k + 3: 10 * (k + 1)])
                ref_inpt.append(_ref_inpt)
                ref_tgt.append(_ref_tgt)

            if num_markables == 0:
                ref_inpt = None
                ref_tgt = None
            else:
                ref_inpt = torch.Tensor(ref_inpt).long()
                ref_tgt = torch.Tensor(ref_tgt).long()

            # construct tensor for selection target
            sel_tgt = torch.Tensor(sels).long()
            if device is not None:
                ctx = ctx.to(device)
                data = data.to(device)
                sel_tgt = sel_tgt.to(device)

            # construct tensor for input and target
            inpt = data.narrow(0, 0, data.size(0) - 1)
            tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

            sel_idxs = torch.Tensor(sel_idxs).long()

            batches.append((ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt, scenario_ids, real_ids, agents, chat_ids, sel_idxs))

        if shuffle:
            random.shuffle(batches)

        return batches, stats

class MarkableCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, domain, path, freq_cutoff=2, train='train_markable.txt',
                 valid='valid_markable.txt', test='test_markable.txt', verbose=False, word_dict=None):
        self.verbose = verbose
        if word_dict is None:
            self.word_dict = Dictionary.from_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff)
        else:
            self.word_dict = word_dict

        self.bio_dict = {"B":0, "I":1, "O":2, "<START>":3, "<STOP>":4, "<PAD>": 5}

        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[1]) for x in self.train])

    def tokenize(self, file_name):
        """Tokenizes the file and produces a dataset."""
        lines = read_lines(file_name)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_vals = [float(val) for val in get_tag(tokens, 'input')]
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))
            markable_idxs = [self.bio_dict[val] for val in get_tag(tokens, 'markables')]
            scenario_id = get_tag(tokens, 'scenario_id')[0]
            agent = int(get_tag(tokens, 'agent')[0])
            chat_id = get_tag(tokens, 'chat_id')[0]
            dataset.append((input_vals, word_idxs, markable_idxs, scenario_id, agent, chat_id))
            # compute statistics
            total += len(word_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.copy(self.train), bsz, shuffle=shuffle, name="train")

    def valid_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.valid), bsz, shuffle=shuffle, name="valid")

    def test_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle, name="test")

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        """Splits given dataset into batches."""
        if shuffle:
            random.shuffle(dataset)

        # sort by dialog length and pad
        dataset.sort(key=lambda x: len(x[1]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0,
        }

        for i in range(0, len(dataset), bsz):
            inputs, words, markables, scenario_ids, agents, chat_ids = [], [], [], [], [], []
            for j in range(i, min(i + bsz, len(dataset))):
                inputs.append(dataset[j][0])
                words.append(dataset[j][1])
                markables.append(dataset[j][2])
                scenario_ids.append(dataset[j][3])
                agents.append(dataset[j][4])
                chat_ids.append(dataset[j][5])
                assert len(words) == len(markables)

            # the longest dialogue in the batch
            max_len = len(words[-1])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                # one additional pad
                words[j] += [pad] * (max_len - len(words[j]) + 1)
                markables[j] += [self.bio_dict["<PAD>"]] * (max_len - len(markables[j]) + 1)

            # construct tensor for context
            ctx = torch.Tensor(inputs).float().squeeze()
            words = torch.Tensor(words).long().transpose(0, 1).contiguous().squeeze()
            markables = torch.Tensor(markables).long().transpose(0, 1).contiguous().squeeze()

            if device is not None:
                ctx = ctx.to(device)
                words = words.to(device)
                markables = markables.to(device)

            batches.append((ctx, words, markables, scenario_ids, agents, chat_ids))

        if shuffle:
            random.shuffle(batches)

        return batches, stats