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

dialogue_tokens = []

vocab = Counter()

corenlp_parser = CoreNLPParser(url='http://localhost:9000')

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

def create_referents(text, markables, referent_annotation, kb, agent):

	referent_tokens = []

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

	#for i in range(len(token2start) - 1):
	#	print(token2start[i])
	#	print(text[token2start[i]:token2start[i+1]])

	assert len(token2start) == len(dialogue_tokens)

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
					break

			for i in range(len(token2start)):
				if i == len(token2start) - 1 or end <= token2start[i+1]:
					end_tok = i
					break

			for i in range(len(token2start)):
				if start <= token2start[i] and dialogue_tokens[i] in ["<eos>", "<selection>"]:
					end_of_utterance_tok = i
					break

			# fix mistake due to tokenization mistaket
			if start_tok > end_tok:
				if chat_id == "C_b07ee61970504668a7b63c3973936129" and markable_id == "M4":
					start_tok = 29
					end_tok = 29
				elif chat_id == "C_22b80a8d7a6e417da0de423aa9bad760":
					start_tok = 15
					end_tok = 15

			# use end of dialogue token
			#end_of_utterance_tok = len(dialogue_tokens) - 1

			#print(text[start:end])
			#print(dialogue_tokens[start_tok:end_tok+1])
			#print(dialogue_tokens[end_of_utterance_tok])

			referent_tokens.append(str(start_tok))
			referent_tokens.append(str(end_tok))
			referent_tokens.append(str(end_of_utterance_tok))

			for ent in kb:
				if "agent_{}_{}".format(agent, ent['id']) in referent_annotation[markable_id]["referents"]:
					referent_tokens.append("1")
				else:
					referent_tokens.append("0")

	return ['<referents>'] + referent_tokens + ['</referents>']

def create_output(kb, events, agent):
	ids = []
	for obj in kb:
		ids.append(obj['id'])
	select_id = None
	for event in events:
		if event['action'] == 'select' and agent == event['agent']:
			select_id = ids.index(event['data'])
	return ['<output>', str(select_id), '</output>']

def create_real_ids(kb):
	real_ids = []
	for obj in kb:
		real_ids.append(obj['id'])
	return ['<real_ids>'] + real_ids + ['</real_ids>']

def create_scenario_id(scenario_id):
	return ['<scenario_id>'] + [scenario_id] + ['</scenario_id>']

def create_agent(agent):
	return ['<agent>'] + [str(agent)] + ['</agent>']

def create_chat_id(chat_id):
	return ['<chat_id>'] + [chat_id] + ['</chat_id>']

def create_markables(markables):
	markable_tokens = []
	for markable in markables:
		markable_id = markable["markable_id"]
		markable_text = markable["text"]
		markable_tokens.append(markable_id)
		markable_tokens.append(markable_text)
	return ['<markables>'] + markable_tokens + ['</markables>']

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--color_range', type=int, default=150, help='range of color')
	parser.add_argument('--size_range', type=int, default=6, help='range of size')
	parser.add_argument('--base_size', type=int, default=10, help='base of size')
	parser.add_argument('--normalize', action='store_true', default=False)
	parser.add_argument('--valid_proportion', type=float, default=0.1)
	parser.add_argument('--test_proportion', type=float, default=0.1)
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

	output_file_names = ['train_reference_' + str(args.seed), 'valid_reference_' + str(args.seed), 'test_reference_' + str(args.seed)]

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
					tokens += create_referents(markable_annotation[chat_id]["text"], markable_annotation[chat_id]["markables"], aggregated_referent_annotation[chat_id], chat['scenario']['kbs'][agent], agent)
					tokens += create_output(chat['scenario']['kbs'][agent], chat['events'], agent)
					tokens += create_real_ids(chat['scenario']['kbs'][agent])
					tokens += create_scenario_id(chat['scenario']['uuid'])
					tokens += create_agent(agent)
					tokens += create_chat_id(chat['uuid'])
					#tokens += create_markables(markable_annotation[chat_id]["markables"])
					out.write(" ".join(tokens) + "\n")
