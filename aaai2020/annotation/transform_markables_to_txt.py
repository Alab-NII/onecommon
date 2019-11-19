import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter

from nltk import word_tokenize, pos_tag
from nltk.parse import CoreNLPParser, CoreNLPDependencyParser
import pdb
import numpy as np
from tqdm import tqdm

from misspelling_dict import misspelling_dict
from replacement_dict import replacement_dict
from normalization_dict import normalization_dict
from morphology_dict import morphology_dict

dialogue_tokens = []

inv_normalization_dict = {}
for k, vs in normalization_dict.items():
	for v in vs:
		inv_normalization_dict[v] = k

replaced_strings = 0
normalized_vocabs = 0
split_morpheme = 0

vocab = Counter()
pos = Counter()
noun_phrase = Counter()
misspellings = Counter()

corenlp_parser = CoreNLPParser(url='http://localhost:9000')
#pos_tagger = CoreNLPParser(url='http://localhost:9001', tagtype='pos')

def is_annotatable_markable(markable):
    if markable["generic"] or markable["no-referent"] or markable["all-referents"] or markable["anaphora"] or markable["cataphora"] or markable["predicative"]:
        return False
    else:
        return True

class Tags:
	@classmethod
	def Input(x):
		return ["<input>"] + x + ["</input>"]
	Context = "input"
	Dialogue = "dialogue"
	Output = "output"
	PartnerContext = "partner_input"

def read_json(path):
    try:
        return json.load(open(path))
    except:
        raise Exception('Error reading JSON from %s' % path)

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

def traverse(t):
    try:
        t.label()
    except AttributeError:
          return
    else:
        if t.label() == 'NP':
            print('NP:'+str(t.leaves()))
            print('NPhead:'+str(t.leaves()[-1]))
            for child in t:
                 traverse(child)
        else:
            for child in t:
                traverse(child)

#def extractNounPhrase(tree):


def normalize_val(val, base_val, val_range):
	normalized_val = (val - base_val) / (val_range / 2)
	assert normalized_val >= -1 and normalized_val <= 1
	return normalized_val

def create_input(kb, args):
	input_tokens = []
	xs = []
	ys = []
	sizes = []
	colors = []
	for obj in kb:
		colors.append(int(re.search(r"[\d]+", obj['color']).group(0)))
		xs.append(obj['x'])
		ys.append(obj['y'])
		sizes.append(obj['size'])
		if args.normalize:
			colors[-1] = normalize_val(colors[-1], args.base_color, args.color_range)
			xs[-1] = normalize_val(xs[-1], args.svg_radius + args.margin, args.svg_radius * 2)
			ys[-1] = normalize_val(ys[-1], args.svg_radius + args.margin, args.svg_radius * 2)
			sizes[-1] = normalize_val(sizes[-1], args.base_size, args.size_range)
		input_tokens += [str(xs[-1]), str(ys[-1]), str(sizes[-1]), str(colors[-1])]
	return ['<input>'] + input_tokens + ['</input>']

def create_dialogue(text, agent):
	global misspellings
	global replaced_strings
	global normalized_vocabs
	global inv_normalization_dict
	global split_morpheme
	global vocab
	global corenlp_parser
	global dialogue_tokens

	dialogue_tokens = []

	for utterance in text.split("\n"):
		if utterance.startswith("{}: ".format(agent)):
			utterance_tokens = ["YOU:"]
		else:
			utterance_tokens = ["THEM:"]

		utterance_string = utterance[3:]

		word_toks = word_tokenize(utterance_string.lower())

		vocab.update(word_toks)

		utterance_tokens += word_toks

		utterance_tokens.append("<eos>")

		dialogue_tokens += utterance_tokens

	# remove last <eos>
	dialogue_tokens = dialogue_tokens[:-1] + ['<selection>']

	return ['<dialogue>'] + dialogue_tokens + ['</dialogue>']

def create_scenario_id(scenario_id):
	return ['<scenario_id>'] + [scenario_id] + ['</scenario_id>']

def create_agent(agent):
	return ['<agent>'] + [str(agent)] + ['</agent>']

def create_chat_id(chat_id):
	return ['<chat_id>'] + [chat_id] + ['</chat_id>']

def create_markables(text, markables, referent_annotation, kb, agent):
	markable_labels = []

	# map: tokens in output dialogue -> starting position in text
	token2start = []
	text_start_pos = 0

	for utterance in text.split("\n"): 
		utterance_start_pos = 0
		token2start.append(text_start_pos + utterance_start_pos)
		utterance_start_pos += len("0: ")
		utterance_string = utterance[utterance_start_pos:]

		word_toks = word_tokenize(utterance_string.lower())

		for word_tok in word_toks:
			token2start.append(text_start_pos + utterance_start_pos)
			utterance_start_pos += len(word_tok)
			while (text_start_pos + utterance_start_pos) < len(text) and text[text_start_pos + utterance_start_pos] == " ":
				utterance_start_pos += 1

		token2start.append(text_start_pos + utterance_start_pos)
		text_start_pos += len(utterance) + 1

	assert len(token2start) == len(dialogue_tokens)

	markable_labels = ["O"] * len(token2start)

	for markable in markables:
		markable_id = markable["markable_id"]
		if markable_id in referent_annotation and markable["speaker"] == agent:
			if "unidentifiable" in referent_annotation[markable_id] and referent_annotation[markable_id]["unidentifiable"]:
				continue
			markable_id = markable["markable_id"]
			start = markable["start"]
			end = markable["end"]

			for i in range(len(token2start)):
				if i == len(token2start) - 1 or start <= token2start[i]:
					start_tok = i
					markable_labels[i] = "B"
					break

			for i in range(len(token2start)):
				if i == len(token2start) - 1 or end <= token2start[i+1]:
					end_tok = i
					break

			for i in range(start_tok + 1, end_tok + 1):
				markable_labels[i] = "I"

	return ['<markables>'] + markable_labels + ['</markables>']

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--color_range', type=int, default=150, help='range of color')
	parser.add_argument('--size_range', type=int, default=6, help='range of size')
	parser.add_argument('--base_size', type=int, default=10, help='base of size')
	parser.add_argument('--normalize', action='store_true', default=False)
	parser.add_argument('--correct_misspellings', action='store_true', default=False)
	parser.add_argument('--replace_strings', action='store_true', default=False)
	parser.add_argument('--normalize_vocab', action='store_true', default=False)
	parser.add_argument('--split_morpheme', action='store_true', default=False)
	parser.add_argument('--valid_proportion', type=float, default=0.1)
	parser.add_argument('--test_proportion', type=float, default=0.1)
	parser.add_argument('--uncorrelated', action='store_true', default=False)
	parser.add_argument('--success_only', action='store_true', default=False)
	parser.add_argument('--seed', type=int, default=1, help='range of size')
	args = parser.parse_args()

	# current support
	args.base_color = 128
	args.svg_radius = 200
	args.margin = 15
	
	np.random.seed(args.seed)

	args.train_proportion = 1 - args.valid_proportion - args.test_proportion

	dialogue_corpus = read_json("final_transcripts.json")
	markable_annotation = read_json("markable_annotation.json")
	aggregated_referent_annotation = read_json("aggregated_referent_annotation.json")
	
	chat_ids = list(aggregated_referent_annotation.keys())

	# shuffle corpus
	np.random.shuffle(chat_ids)

	total_size = len(chat_ids)
	split_index = [0, int(total_size * args.train_proportion),
					int(total_size * (args.train_proportion + args.valid_proportion)), -1]
	tags = Tags()

	output_file_names = ['train_markable_' + str(args.seed), 'valid_markable_' + str(args.seed), 'test_markable_' + str(args.seed)]

	for i, output_file in enumerate(output_file_names):
		with open('{}.txt'.format(output_file), 'w') as out:
			start = split_index[i]
			end = split_index[i+1]
			for chat_id in tqdm(chat_ids[start:end]):	
				chat = [chat for chat in dialogue_corpus if chat['uuid'] == chat_id]
				chat = chat[0]
				for agent in [0,1]:
					tokens = []
					tokens += create_input(chat['scenario']['kbs'][agent], args)
					tokens += create_dialogue(markable_annotation[chat_id]["text"], agent)
					tokens += create_markables(markable_annotation[chat_id]["text"], markable_annotation[chat_id]["markables"], aggregated_referent_annotation[chat_id], chat['scenario']['kbs'][agent], agent)
					tokens += create_scenario_id(chat['scenario']['uuid'])
					tokens += create_agent(agent)
					tokens += create_chat_id(chat['uuid'])
					out.write(" ".join(tokens) + "\n")

	pdb.set_trace()