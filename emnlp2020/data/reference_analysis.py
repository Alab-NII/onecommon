import argparse
import copy
import json
import os
import pickle
import re
import sys
import traceback
from collections import Counter, defaultdict
import glob
import itertools

from nltk import word_tokenize, pos_tag, bigrams
from nltk.parse import CoreNLPParser, CoreNLPDependencyParser
from mosestokenizer import MosesTokenizer, MosesDetokenizer
import editdistance

import pdb
import numpy as np

from tqdm import tqdm

from minimal_misspelling_dict import misspelling_dict
from minimal_replacement_dict import replacement_dict

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
sns.set(font_scale=1.15)

vocab = Counter()
pos = Counter()
noun_phrase = Counter()

fixed_misspellings = 0
replaced_strings = 0
normalized_vocabs = 0

PUNCTUATION = ['’', '_', "'", '<', '*', '\\', '$', '%', '"', ')', '+', '.', '(', '!', ',', ']', '[', '@', '~', '#', ':', '&', ' ', '>', '-', '=', ';', '/', '?']

# create new misspelling dictionary for minimal preprocessing
minimal_misspelling_dict = {}
for misspelling, correct_spelling in misspelling_dict.items():
	if isinstance(correct_spelling, str):
		minimal_misspelling_dict[misspelling] = correct_spelling
		minimal_misspelling_dict[misspelling.upper()] = correct_spelling.upper()
		minimal_misspelling_dict[misspelling.capitalize()] = correct_spelling.capitalize()
	else:
		minimal_misspelling_dict[misspelling] = " ".join(correct_spelling)
		minimal_misspelling_dict[misspelling.upper()] = " ".join(correct_spelling).upper()
		minimal_misspelling_dict[misspelling.capitalize()] = " ".join(correct_spelling).capitalize()

#DETOKENIZE = [".", ",", "?", "!", "'s", "n't", "'ve", "'re", "'m", "'ll", "'d"] # TODO: make sure this is complete

def is_annotatable_markable(markable):
    if markable["generic"] or markable["no-referent"] or markable["all-referents"] or (markable["anaphora"] is not None) or (markable["cataphora"] is not None) or (markable["predicative"] is not None):
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

class Markable:
	def __init__(self, markable_id, label, text, start, end):
		self.markable_id = markable_id
		self.label = label # "Markable" or "None"
		self.text = text
		self.start = start
		self.end = end

		# attributes
		self.no_referent = False
		self.all_referents = False
		self.generic = False

		# relations
		self.anaphora = None
		self.cataphora = None
		self.predicative = None

		self.fixed_text = None
		self.speaker = None

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

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

def plot_referent_color(args, dialogue_corpus):
	markable_annotation = read_json("markable_annotation.json")
	referent_annotation = read_json(args.referent_annotation)
	aggregated_referent_annotation = read_json("aggregated_referent_annotation.json")
	chat_ids = list(referent_annotation.keys())

	text2color = defaultdict(list)

	for dialogue in dialogue_corpus:
		chat_id = dialogue["uuid"]
		if not chat_id in markable_annotation:
			continue
		else:
			markables = markable_annotation[chat_id]
		if not chat_id in aggregated_referent_annotation:
			continue

		agent_0_kb, agent_1_kb = dialogue["scenario"]["kbs"]
		ent_id2color = {x['id'] : int(x['color'].split(',')[1]) for x in agent_0_kb + agent_1_kb}

		for markable in markables["markables"]:
			markable_id = markable["markable_id"]
			speaker = markable["speaker"]
			if markable_id not in aggregated_referent_annotation[chat_id]:
				continue
			if not "unidentifiable" in aggregated_referent_annotation[chat_id][markable_id] or not aggregated_referent_annotation[chat_id][markable_id]["unidentifiable"]:
				for referent in referent_annotation[chat_id][markable_id]["referents"]:
					referent_id = referent.split("_")[2]
					referent_color = ent_id2color[referent_id]
					markable_unigrams = [w.lower() for w in word_tokenize(markable["text"])]
					markable_bigrams = list((bigrams(markable_unigrams)))
					if "black" in markable_unigrams:
						text2color["black"].append(referent_color)
					elif ("very", "dark") in markable_bigrams \
						or ("extremely", "dark") in markable_bigrams \
						or ("super", "dark") in markable_bigrams \
						or ("really", "dark") in markable_bigrams:
						text2color["very dark"].append(referent_color)
					elif ("very", "light") in markable_bigrams \
						or ("extremely", "light") in markable_bigrams \
						or ("super", "light") in markable_bigrams \
						or ("really", "light") in markable_bigrams:
						text2color["very light"].append(referent_color)
					elif ("medium", "gray") in markable_bigrams \
						or ("medium", "grey") in markable_bigrams \
						or ("medium", "dark") in markable_bigrams \
						or ("medium", "shade") in markable_bigrams \
						or ("medium", "colored") in markable_bigrams \
						or ("med", "gray") in markable_bigrams \
						or ("med", "grey") in markable_bigrams \
						or ("med", "dark") in markable_bigrams:
						text2color["medium"].append(referent_color)
					elif "light" in markable_unigrams:
						text2color["light"].append(referent_color)
					elif "dark" in markable_unigrams:
						text2color["dark"].append(referent_color)
					elif "darker" in markable_unigrams:
						text2color["darker"].append(referent_color)
					elif "darkest" in markable_unigrams:
						text2color["darkest"].append(referent_color)
					elif "lighter" in markable_unigrams:
						text2color["lighter"].append(referent_color)
					elif "lightest" in markable_unigrams:
						text2color["lightest"].append(referent_color)
					elif "gray" in markable_unigrams \
						or "grey" in markable_unigrams:
						text2color["gray"].append(referent_color)

	plot_colors = sns.color_palette("hls", 11)
	for i, color in enumerate(["black", "very dark", "darkest", "dark", "darker", "medium", "gray", "lighter", "light", "lightest", "very light"]):
		print("{}: mean {}, std {} (total {})".format(color, np.mean(text2color[color]), np.std(text2color[color]), len(text2color[color])))
		sns.distplot(text2color[color], hist=False, color=plot_colors[i], label=color)
	plt.xlabel('color', fontsize=16)
	plt.ylabel('probability density', fontsize=16)
	plt.legend(fontsize='x-small')
	plt.tight_layout()
	plt.savefig(args.referent_annotation.split(".")[0] + '_color.png', dpi=400)
	plt.clf()


def plot_referent_size(args, dialogue_corpus):
	markable_annotation = read_json("markable_annotation.json")
	referent_annotation = read_json(args.referent_annotation)
	chat_ids = list(referent_annotation.keys())

	text2size = defaultdict(list)

	for dialogue in dialogue_corpus:
		chat_id = dialogue["uuid"]
		if not chat_id in markable_annotation:
			continue
		else:
			markables = markable_annotation[chat_id]
		if not chat_id in referent_annotation:
			continue

		agent_0_kb, agent_1_kb = dialogue["scenario"]["kbs"]
		ent_id2size = {x['id'] : x['size'] for x in agent_0_kb + agent_1_kb}

		for markable in markables["markables"]:
			markable_id = markable["markable_id"]
			speaker = markable["speaker"]
			if markable_id not in referent_annotation[chat_id]:
				continue
			if not "unidentifiable" in referent_annotation[chat_id][markable_id] or not referent_annotation[chat_id][markable_id]["unidentifiable"]:
				for referent in referent_annotation[chat_id][markable_id]["referents"]:
					referent_id = referent.split("_")[2]
					referent_size = ent_id2size[referent_id]
					markable_unigrams = [w.lower() for w in word_tokenize(markable["text"])]
					markable_bigrams = list((bigrams(markable_unigrams)))
					if "largest" in markable_unigrams \
						or "biggest" in markable_unigrams:
						text2size["largest"].append(referent_size)
					if "larger" in markable_unigrams \
						or "bigger" in markable_unigrams:
						text2size["larger"].append(referent_size)
					if ("very", "large") in markable_bigrams \
						or ("very", "big") in markable_bigrams \
						or ("extremely", "large") in markable_bigrams \
						or ("extremely", "big") in markable_bigrams:
						text2size["very large"].append(referent_size)
					elif ("slightly", "large") in markable_bigrams \
						or ("slightly", "big") in markable_bigrams \
						or ("bit", "big") in markable_bigrams \
						or ("bit", "large") in markable_bigrams \
						or ("subtly", "large") in markable_bigrams \
						or ("subtly", "big") in markable_bigrams:
						text2size["slightly large"].append(referent_size)
					elif "large" in markable_unigrams \
						or "big" in markable_unigrams:
						text2size["large"].append(referent_size)
					if ("medium", "size") in markable_bigrams \
						or ("medium", "sized") in markable_bigrams \
						or ("med", "size") in markable_bigrams \
						or ("med", "sized") in markable_bigrams:
						text2size["medium"].append(referent_size)
					if ("very", "small") in markable_bigrams \
						or ("very", "tiny") in markable_bigrams \
						or ("extremely", "small") in markable_bigrams \
						or ("extremely", "tiny") in markable_bigrams:
						text2size["very small"].append(referent_size)
					elif ("slightly", "small") in markable_bigrams \
						or ("slightly", "tiny") in markable_bigrams \
						or ("bit", "small") in markable_bigrams \
						or ("bit", "tiny") in markable_bigrams:
						text2size["slightly small"].append(referent_size)
					elif "small" in markable_unigrams \
						or "tiny" in markable_unigrams:
						text2size["small"].append(referent_size)
					if "smaller" in markable_unigrams \
						or "tinier" in markable_unigrams:
						text2size["smaller"].append(referent_size)
					if "smallest" in markable_unigrams \
						or "tiniest" in markable_unigrams:
						text2size["smallest"].append(referent_size)

	plot_colors = sns.color_palette("hls", 9)
	x = list(range(7, 14))
	for i, size in enumerate(['very small', 'smallest', 'small', 'smaller', 'medium', 'larger', 'large', 'largest', 'very large']):
		print("{}: mean {}, std {} (total {})".format(size, np.mean(text2size[size]), np.std(text2size[size]), len(text2size[size])))
		y = []
		for size_val in x:
			y.append(text2size[size].count(size_val))
		y = np.divide(y, len(text2size[size]))
		sns.lineplot(x=x, y=y, color=plot_colors[i], label=size)
		#sns.distplot(text2size[size], kde=False, norm_hist=True, color=plot_colors[i], label=size)
	plt.xlabel('size', fontsize=16)
	plt.ylabel('probability density', fontsize=16)
	plt.legend(fontsize='x-small')
	plt.tight_layout()
	plt.savefig(args.referent_annotation.split(".")[0] + '_size.png', dpi=400)
	plt.clf()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--scenario_file', type=str, default="aaai_train_scenarios.json")
	parser.add_argument('--scenario_file_2', type=str, default="aaai_train_scenarios_2.json")
	parser.add_argument('--transcript_file', type=str, default="final_transcripts.json")

	parser.add_argument('--plot_referent_color', action='store_true', default=False)
	parser.add_argument('--plot_referent_size', action='store_true', default=False)
	parser.add_argument('--referent_annotation', type=str, default="aggregated_referent_annotation.json")

	args = parser.parse_args()

	dialogue_corpus = read_json(args.transcript_file)
	scenario_list = read_json(args.scenario_file)
	scenario_list += read_json(args.scenario_file_2)

	if args.plot_referent_color:
		plot_referent_color(args, dialogue_corpus)

	if args.plot_referent_size:
		plot_referent_size(args, dialogue_corpus)

