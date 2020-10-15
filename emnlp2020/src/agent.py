import sys
from collections import defaultdict
import pdb

import numpy as np
import torch
from torch import optim, autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from dialog import DialogLogger
import domain
from engines import Criterion
import math
from collections import Counter

from nltk.parse import CoreNLPParser, CoreNLPDependencyParser



class Agent(object):
    """ Agent's interface. """
    def feed_context(self, ctx):
        pass

    def read(self, inpt):
        pass

    def write(self):
        pass

    def choose(self):
        pass

    def update(self, agree, reward, choice):
        pass

    def get_attention(self):
        return None


class RnnAgent(Agent):
    def __init__(self, model, args, name='Alice', train=False):
        super(RnnAgent, self).__init__()
        self.model = model
        self.args = args
        self.name = name
        self.human = False
        self.domain = domain.get_domain(args.domain)
        self.train = train
        if train:
            self.model.train()
            self.opt = optim.RMSprop(
            self.model.parameters(),
            lr=args.rl_lr,
            momentum=self.args.momentum)
            self.all_rewards = []
            self.t = 0
        else:
            self.model.eval()

    def _encode(self, inpt, dictionary):
        encoded = torch.Tensor(dictionary.w2i(inpt)).long().unsqueeze(1)
        return encoded

    def _decode(self, out, dictionary):
        return dictionary.i2w(out.data.squeeze(1).cpu())

    def feed_context(self, context):
        self.lang_hs = []
        self.logprobs = []
        self.sents = []
        self.words = []
        self.context = context
        self.ctx = torch.Tensor([float(x) for x in context]).float().unsqueeze(1)
        self.ctx_h = self.model.ctx_encoder(Variable(self.ctx))
        self.lang_h = self.model._zero(1, self.model.args.nhid_lang)

    def feed_partner_context(self, partner_context):
        pass

    def read(self, inpt):
        self.sents.append(Variable(self._encode(['THEM:'] + inpt, self.model.word_dict)))
        inpt = self._encode(inpt, self.model.word_dict)
        lang_hs, self.lang_h = self.model.read(self.ctx_h, Variable(inpt), self.lang_h.unsqueeze(0))
        self.lang_h = self.lang_h.squeeze(0)
        self.lang_hs.append(lang_hs.squeeze(1))
        self.words.append(self.model.word2var('THEM:').unsqueeze(0))
        self.words.append(Variable(inpt))
        #assert (torch.cat(self.words).size(0) == torch.cat(self.lang_hs).size(0))

    def write(self, max_words=100):
        outs, logprobs, self.lang_h, lang_hs = self.model.write(self.ctx_h, self.lang_h, 
                                                            max_words, self.args.temperature)
        self.logprobs.extend(logprobs)
        self.lang_hs.append(lang_hs)
        #self.words.append(self.model.word2var('YOU:').unsqueeze(0))
        self.words.append(outs)
        self.sents.append(torch.cat([self.model.word2var('YOU:').unsqueeze(1), outs], 0))

        """if self.args.visualize_referents:
            #utterance = self._decode(outs, self.model.word_dict)[1:-1]
            #const_tree = list(self.corenlp_parser.parse(utterance))
            utterance = self._decode(outs, self.model.word_dict)
            ref_inpt = [3, 6, len(utterance) - 1]
            ref_inpt = torch.Tensor(ref_inpt).long().unsqueeze(0).unsqueeze(0)
            ref_out = self.model.reference_resolution(self.ctx_h, lang_hs.unsqueeze(1), ref_inpt)
            pdb.set_trace()"""

        #if not (torch.cat(self.words).size(0) + 1 == torch.cat(self.lang_hs).size(0)):
        #    pdb.set_trace()
        #assert (torch.cat(self.words).size(0) + 1 == torch.cat(self.lang_hs).size(0))
        # remove 'YOU:'
        outs = outs.narrow(0, 1, outs.size(0) - 1)
        return self._decode(outs, self.model.word_dict)

    def predict_referents(self, ref_inpt):
        if len(ref_inpt) == 0:
            ref_inpt = None
        else:
            ref_inpt = torch.Tensor(ref_inpt).long().unsqueeze(0)
        ref_out = self.model.reference_resolution(self.ctx_h, torch.cat(self.lang_hs, 0).unsqueeze(1), ref_inpt)
        if ref_out is not None:
            return ref_out.squeeze(1)
        else:
            return None

    def _make_idxs(self, sents):
        lens, rev_idxs, hid_idxs = [], [], []
        for sent in sents:
            assert sent.size(1) == 1
            # remove last hidden state
            ln = torch.Tensor(1).fill_(sent.size(0) - 1).long()
            lens.append(ln)
            idx = torch.Tensor(sent.size(0), 1, 1).fill_(-1).long()
            for j in range(idx.size(0)):
                idx[j][0][0] = j if j >= sent.size(0) else sent.size(0) - j - 1
            rev_idxs.append(Variable(idx))
            hid_idxs.append(Variable(ln.view(1, 1, 1)))
        return lens, rev_idxs, hid_idxs

    def _choose(self, sample=False):
        outs_emb = torch.cat(self.lang_hs).unsqueeze(1)
        sel_idx = torch.Tensor(1).fill_(torch.cat(self.lang_hs).size(0) - 1).long()
        choice_logit = self.model.selection(self.ctx_h, outs_emb, sel_idx)

        prob = F.softmax(choice_logit, dim=1)
        if sample:
            idx = prob.multinomial(1).detach()
            logprob = F.log_softmax(choice_logit, dim=1).gather(1, idx)
        else:
            _, idx = prob.max(1, keepdim=True)
            logprob = None

        # Pick only your choice
        return idx.item(), prob.gather(1, idx), logprob

    def choose(self):
        if self.args.eps < np.random.rand():
            choice, _, _ = self._choose(sample=False)
        else:
            choice, _, logprob = self._choose(sample=True)
            self.logprobs.append(logprob)

        choice, _, _ = self._choose()
        if self.real_ids:
            choice = self.real_ids[choice]
        return choice

    def update(self, agree, reward, choice=None):
        if not self.train:
            return

        self.t += 1
        if len(self.logprobs) == 0:
            return

        self.all_rewards.append(reward)

        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        g = Variable(torch.zeros(1, 1).fill_(r))
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.rl_clip)
        if self.args.visual and self.t % 10 == 0:
            self.model_plot.update(self.t)
            self.agree_plot.update('agree', self.t, int(agree))
            self.reward_plot.update('reward', self.t, reward)
            self.reward_plot.update('partner_reward', self.t, partner_reward)
            self.agree_reward_plot.update('reward', self.t, reward_agree)
            self.agree_reward_plot.update('partner_reward', self.t, partner_reward_agree)
            self.loss_plot.update('loss', self.t, loss.data[0][0])

        self.opt.step()

