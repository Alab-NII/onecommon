import sys
import pdb

import numpy as np

import torch

from metric import MetricsContainer
import data
import utils
import domain

class DialogLogger(object):
    def __init__(self, verbose=False, log_file=None, append=False, scenarios=None):
        self.logs = []
        if verbose:
            self.logs.append(sys.stderr)
        if log_file:
            flags = 'a' if append else 'w'
            self.logs.append(open(log_file, flags))
        
        self.scenarios = scenarios

    def _dump(self, s, forced=False):
        for log in self.logs:
            print(s, file=log)
            log.flush()
        if forced:
            print(s, file=sys.stdout)
            sys.stdout.flush()

    def _dump_with_name(self, name, s):
        self._dump('{0: <5} : {1}'.format(name, s))

    def _scenario_to_svg(self, scenario, choice=None):
        svg_list = []
        for agent in [0,1]:
            svg = "<svg width=\"{0}\" height=\"{0}\" id=\"{1}\">".format(430, "agent_" + str(agent))
            svg += '''<circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>'''
            for obj in scenario['kbs'][agent]:
                svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"{3}\"/>".format(obj['x'], obj['y'], 
                    obj['size'], obj['color'])
                if choice and choice[agent] == obj['id']:
                    if agent == 0:
                        agent_color = "red"
                    else:
                        agent_color = "blue"
                    svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"none\" stroke=\"{3}\" stroke-width=\"4\" stroke-dasharray=\"3,3\" />".format(obj['x'], obj['y'],
                        obj['size'] + 4, agent_color)
            svg += "</svg>"
            svg_list.append(svg)
        return svg_list

    def _attention_to_svg(self, scenario, agent, attention=None):
        svg = '''<svg id="svg" width="430" height="430"><circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/> '''
        for obj, attention_weight in zip(scenario['kbs'][agent], attention):
            svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"rgb(255,{3},{3})\" />".format(obj['x'], obj['y'],
                                                                                                 obj['size'], int((1 - attention_weight) * 255))
        svg += '''</svg>'''
        return svg

    def dump_sent(self, name, sent):
        self._dump_with_name(name, ' '.join(sent))

    def dump_attention(self, agent_name, agent_id, scenario_id, attention):
        svg = self._attention_to_svg(self.scenarios[scenario_id], agent_id, attention)
        self._dump_with_name('%s_attention' % agent_name, svg)

    def dump_choice(self, scenario_id, choice):
        self._dump_with_name('scenario_id', scenario_id)
        svg_list = self._scenario_to_svg(self.scenarios[scenario_id], choice)
        self._dump_with_name('Alice', svg_list[0])
        self._dump_with_name('Bob', svg_list[1])

    def dump_agreement(self, agree):
        self._dump('Agreement!' if agree else 'Disagreement?!')

    def dump(self, s, forced=False):
        self._dump(s, forced=forced)


class DialogSelfTrainLogger(DialogLogger):
    def __init__(self, verbose=False, log_file=None):
        super(DialogSelfTrainLogger, self).__init__(verbose, log_file)
        self.name2example = {}
        self.name2choice = {}

    def _dump_with_name(self, name, sent):
        for n in self.name2example:
            if n == name:
                self.name2example[n] += " YOU: "
            else:
                self.name2example[n] += " THEM: "

            self.name2example[n] += sent

    def dump_ctx(self, name, ctx):
        self.name2example[name] = ' '.join(ctx)

    def dump_choice(self, name, choice):
        self.name2choice[name] = ' '.join(choice)

    def dump_agreement(self, agree):
        if agree:
            for name in self.name2example:
                for other_name in self.name2example:
                    if name != other_name:
                        self.name2example[name] += ' ' + self.name2choice[name]
                        self.name2example[name] += ' ' + self.name2choice[other_name]
                        self._dump(self.name2example[name])


class Dialog(object):
    def __init__(self, agents, args, markable_detector, markable_detector_corpus):
        # For now we only suppport dialog of 2 agents
        assert len(agents) == 2
        self.agents = agents
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self._register_metrics()
        self.markable_detector = markable_detector
        self.markable_detector_corpus = markable_detector_corpus
        self.selfplay_markables = {}
        self.selfplay_referents = {}

    def _register_metrics(self):
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        self.metrics.register_percentage('agree')
        self.metrics.register_moving_percentage('moving_agree')
        self.metrics.register_average('advantage')
        self.metrics.register_moving_average('moving_advantage')
        self.metrics.register_time('time')
        self.metrics.register_average('comb_rew')
        self.metrics.register_average('agree_comb_rew')
        for agent in self.agents:
            self.metrics.register_average('%s_rew' % agent.name)
            self.metrics.register_moving_average('%s_moving_rew' % agent.name)
            self.metrics.register_average('agree_%s_rew' % agent.name)
            self.metrics.register_percentage('%s_make_sel' % agent.name)
            self.metrics.register_uniqueness('%s_unique' % agent.name)
            if "plot_metrics" in self.args and self.args.plot_metrics:
                self.metrics.register_select_frequency('%s_sel_bias' % agent.name)
        # text metrics
        if self.args.ref_text:
            ref_text = ' '.join(data.read_lines(self.args.ref_text))
            self.metrics.register_ngram('full_match', text=ref_text)

    def _is_selection(self, out):
        return '<selection>' in out

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def plot_metrics(self):
        self.metrics.plot()

    def run(self, ctxs, logger, max_words=5000):
        scenario_id = ctxs[0][0]

        for agent, agent_id, ctx, real_ids in zip(self.agents, [0, 1], ctxs[1], ctxs[2]):
            agent.feed_context(ctx)
            agent.real_ids = real_ids
            agent.agent_id = agent_id

        # Choose who goes first by random
        if np.random.rand() < 0.5:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        conv = []
        speaker = []
        self.metrics.reset()

        words_left = max_words
        length = 0
        expired = False

        while True:
            out = writer.write(max_words=words_left)
            words_left -= len(out)
            length += len(out)

            self.metrics.record('sent_len', len(out))
            if 'full_match' in self.metrics.metrics:
                self.metrics.record('full_match', out)
            self.metrics.record('%s_unique' % writer.name, out)

            conv.append(out)
            speaker.append(writer.agent_id)
            reader.read(out)
            if not writer.human:
                logger.dump_sent(writer.name, out)

            if logger.scenarios and self.args.log_attention:
                attention = writer.get_attention()
                if attention is not None:
                    logger.dump_attention(writer.name, writer.agent_id, scenario_id, attention)

            if self._is_selection(out):
                self.metrics.record('%s_make_sel' % writer.name, 1)
                self.metrics.record('%s_make_sel' % reader.name, 0)
                break

            if words_left <= 1:
                break

            writer, reader = reader, writer

        choices = []
        for agent in self.agents:
            choice = agent.choose()
            choices.append(choice)
        if logger.scenarios:
            logger.dump_choice(scenario_id, choices)
            if "plot_metrics" in self.args and self.args.plot_metrics:
                for agent in [0, 1]:
                    for obj in logger.scenarios[scenario_id]['kbs'][agent]:
                        if obj['id'] == choices[agent]:
                            self.metrics.record('%s_sel_bias' % writer.name, obj,
                                logger.scenarios[scenario_id]['kbs'][agent])

        agree, rewards = self.domain.score_choices(choices, ctxs)
        if expired:
            agree = False
        logger.dump('-' * 80)
        logger.dump_agreement(agree)
        for i, (agent, reward) in enumerate(zip(self.agents, rewards)):
            j = 1 if i == 0 else 0
            agent.update(agree, reward, choice=choices[i])

        if agree:
            self.metrics.record('advantage', rewards[0] - rewards[1])
            self.metrics.record('moving_advantage', rewards[0] - rewards[1])
            self.metrics.record('agree_comb_rew', np.sum(rewards))
            for agent, reward in zip(self.agents, rewards):
                self.metrics.record('agree_%s_rew' % agent.name, reward)

        self.metrics.record('time')
        self.metrics.record('dialog_len', len(conv))
        self.metrics.record('agree', int(agree))
        self.metrics.record('moving_agree', int(agree))
        self.metrics.record('comb_rew', np.sum(rewards) if agree else 0)
        for agent, reward in zip(self.agents, rewards):
            self.metrics.record('%s_rew' % agent.name, reward if agree else 0)
            self.metrics.record('%s_moving_rew' % agent.name, reward if agree else 0)

        if self.markable_detector is not None and self.markable_detector_corpus is not None:
            markable_list = []
            referents_dict = {}

            markable_starts = []
            for agent in [0, 1]:
                dialog_tokens = []
                dialog_text = ""
                markables = []
                for spkr, uttr in zip(speaker, conv):
                    if spkr == agent:
                        dialog_tokens.append("YOU:")
                    else:
                        dialog_tokens.append("THEM:")
                    dialog_tokens += uttr
                    dialog_text += str(spkr) + ": " + " ".join(uttr[:-1]) + "\n"

                    words = self.markable_detector_corpus.word_dict.w2i(dialog_tokens)
                    words = torch.Tensor(words).long().cuda()
                    score, tag_seq = self.markable_detector(words)
                    referent_inpt = []
                    markable_ids = []
                    my_utterance = None
                    current_text = ""
                    for i, word in enumerate(words):
                        if word.item() == self.markable_detector_corpus.word_dict.word2idx["YOU:"]:
                            my_utterance = True
                            current_speaker = agent
                        elif word.item() == self.markable_detector_corpus.word_dict.word2idx["THEM:"]:
                            my_utterance = False
                            current_speaker = 1 - agent
                        if my_utterance:
                            if tag_seq[i].item() == self.markable_detector_corpus.bio_dict["B"]:
                                start_idx = i
                                for j in range(i + 1, len(tag_seq)):
                                    if tag_seq[j].item() != self.markable_detector_corpus.bio_dict["I"]:
                                        end_idx = j - 1
                                        break
                                for j in range(i + 1, len(tag_seq)):
                                    if tag_seq[j].item() in self.markable_detector_corpus.word_dict.w2i(["<eos>", "<selection>"]):
                                        end_uttr = j
                                        break

                                markable_start = len(current_text + " ")
                                if markable_start not in markable_starts:
                                    referent_inpt.append([start_idx, end_idx, end_uttr])
                                    markable_ids.append(len(markable_starts))

                                    # add markable
                                    markable = {}
                                    markable["start"] = markable_start
                                    markable["end"] = len(current_text + " " + " ".join(dialog_tokens[start_idx:end_idx + 1]))
                                    #markable["start"] = len(str(spkr) + ": " + " ".join(dialog_tokens[1:start_idx]) + " ")
                                    #markable["end"] = len(str(spkr) + ": " + " ".join(dialog_tokens[1:end_idx + 1]))
                                    markable["markable_id"] = len(markable_starts)
                                    markable["speaker"] = current_speaker
                                    markable["text"] = " ".join(dialog_tokens[start_idx:end_idx + 1])
                                    markable_starts.append(markable["start"])
                                    markable_list.append(markable)

                        if word.item() == self.markable_detector_corpus.word_dict.word2idx["YOU:"]:
                            current_text += "{}:".format(current_speaker)
                        elif word.item() == self.markable_detector_corpus.word_dict.word2idx["THEM:"]:
                            current_text += "{}:".format(current_speaker)
                        elif word.item() in self.markable_detector_corpus.word_dict.w2i(["<eos>", "<selection>"]):
                            current_text += "\n"
                        else:
                            current_text += " " + self.markable_detector_corpus.word_dict.idx2word[word.item()]

                    assert len(current_text) == len(dialog_text)

                    ref_out = self.agents[agent].predict_referents(referent_inpt)

                    if ref_out is not None:
                        for i, markable_id in enumerate(markable_ids):
                            ent_ids = [ent["id"] for ent in logger.scenarios[scenario_id]['kbs'][agent]]
                            referents = []
                            for j, is_referent in enumerate((ref_out[i] > 0).tolist()):
                                if is_referent:
                                    referents.append("agent_" + str(agent) + "_" + ent_ids[j])

                            referents_dict[markable_id] = referents

            #markable_starts = list(set(markable_starts))
            # reindex markable ids
            markable_id_and_start = [(markable_id, markable_start) for markable_id, markable_start in zip(range(len(markable_starts)), markable_starts)]
            reindexed_markable_ids = [markable_id for markable_id, _ in sorted(markable_id_and_start, key = lambda x: x[1])]

            self.selfplay_markables[scenario_id] = {}
            self.selfplay_referents[scenario_id] = {}

            # add markables
            self.selfplay_markables[scenario_id]["markables"] = []
            for new_markable_id, old_markable_id in enumerate(reindexed_markable_ids):
                markable = markable_list[old_markable_id]
                markable["markable_id"] = "M{}".format(new_markable_id + 1)
                self.selfplay_markables[scenario_id]["markables"].append(markable)

            # add dialogue text
            self.selfplay_markables[scenario_id]["text"] = dialog_text

            # add final selections
            self.selfplay_markables[scenario_id]["selections"] = choices

            # add referents
            for new_markable_id, old_markable_id in enumerate(reindexed_markable_ids):
                referents = referents_dict[old_markable_id]
                self.selfplay_referents[scenario_id]["M{}".format(new_markable_id + 1)] = referents

        logger.dump('-' * 80)
        logger.dump(self.show_metrics())
        logger.dump('-' * 80)
        #for ctx, choice in zip(ctxs, choices):
        #    logger.dump('debug: %s %s' % (' '.join(ctx), ' '.join(choice)))

        return conv, agree, rewards
        