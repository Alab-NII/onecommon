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


def preprocess_utterance(args, utterance_string):
	global replaced_strings
	global fixed_misspellings
	global normalized_vocabs

	if args.replace_strings:
		for key in replacement_dict.keys():
			if key in utterance_string:
				replaced_strings += 1
				utterance_string = utterance_string.replace(key, replacement_dict[key])

		utterance_string = utterance_string.replace("’", "'")
		utterance_string = re.sub(r"\.[\.]+", "...", utterance_string)
		utterance_string = re.sub(r"!![!]+", "!!!", utterance_string)
		utterance_string = re.sub(r"\?\?[\?]+", "???", utterance_string)
		utterance_string = re.sub(r" [ ]+", " ", utterance_string)

	fixed_tokens = []
	if args.correct_misspellings:
		utterance_tokens = utterance_string.split(" ")
		for tok in utterance_tokens:
			if len(word_tokenize(tok)) >= 2:
				sub_toks = word_tokenize(tok)
				fixed_sub_toks = []
				for sub_tok in sub_toks:
					if sub_tok.lower() in minimal_misspelling_dict:
						fixed_misspellings += 1
						if sub_tok in minimal_misspelling_dict:
							fixed_sub_toks.append(minimal_misspelling_dict[sub_tok])
						else:
							fixed_sub_toks.append(minimal_misspelling_dict[sub_tok.lower()])
					else:
						fixed_sub_toks.append(sub_tok)
						vocab[sub_tok] += 1
				tok = "".join(fixed_sub_toks)
				tok = tok.replace("``", "\"")
				tok = tok.replace("''", "\"")
				fixed_tokens.append(tok)
			else:
				if tok.lower() in minimal_misspelling_dict:
					fixed_misspellings += 1
					if tok in minimal_misspelling_dict:
						fixed_tokens.append(minimal_misspelling_dict[tok])
					else:
						fixed_tokens.append(minimal_misspelling_dict[tok.lower()])
				else:
					fixed_tokens.append(tok)
					vocab[tok] += 1

	return " ".join(fixed_tokens).strip()


def detect_mentions(t, unnested_only=False):
	"""
		detect all mentions (noun phrases) in the parsed utterance
	"""
	mentions = []
	try:
		t.label()
	except AttributeError:
		return mentions
	else:
		if t.label() == 'NP':
			nested = False
			for child in t:
				child_mentions = detect_mentions(child, unnested_only)
				if len(child_mentions) > 0:
					nested = True
				mentions += child_mentions

			if unnested_only:
				if not nested:
					mentions.append(t.leaves())
			else:
				mentions.append(t.leaves())
		else:
			for child in t:
				mentions += detect_mentions(child, unnested_only)
		return mentions

def validate_mention(mention):
	"""
		check whether the noun phrase is a (possible) mentions of the entities under consideration
	"""

	VALID_DETERMINERS = ["a", "an", "another", "either", "neither", "each", "every", "the",  "this", "these", "those", "my", "your", "no", "none", "any", "some", "most", "many", "lots", "one", "few", "two", "three", "four", "five", "six", "seven", "1", "2", "3", "4", "5", "6", "7"]

	# empty
	if len(mention) == 0:
		return False

	if len(mention) == 1:
		if mention[-1].lower() in ["it", "they", "them", "mine", "yours", "those", "that", "itself"]: #"this"
			return True
		# include possessives
		elif mention[-1].lower() in ["its", "their"]:
			return True
		elif mention[-1].lower() in ["anything", "something", "none", "nothing"]:
			return True

	# referring to players
	if mention[-1].lower() in ["i", "we", "you", "us"]:
		return False
	# unlikely mention to dots
	if mention[-1].lower() in ["o'clock", "lets", "circle", "edge", "side", "yes", "slightly", "screen", "position", "inch", "here", "luck", "bit", "shade", "size"]:
		return False
	# wrong pos: not a noun
	if mention[-1].lower() in ["kind", "'s", "yes", "same", "ok", "hmm", "let"]:
		return False

	# singular
	if mention[-1].lower() in ["dot", "one", "point"]:
		return True

	# plurals
	if mention[-1].lower() in ["dots", "ones", "points", "two", "three", "four", "five", "six", "seven"]:
		return True

	# grouping
	if mention[-1].lower() in ["line", "triangle", "triangles", "set", "sets", "cluster", "square", "diamond", "trapezoid", "group", "kite", "pair", "couple", "twin", "twins"]:
		return True

	# referring by colors
	if mention[0].lower() in VALID_DETERMINERS and mention[-1].lower() in ["dark", "gray", "grey", "light", "black", "darker", "lighter", "darkest", "blackest"]:
		return True

	# referring by size
	if mention[0].lower() in VALID_DETERMINERS and mention[-1].lower() in ["big", "large", "small", "tiny", "larger", "bigger", "smaller", "tinier", "biggest", "largest", "smallest", "tinest"]:
		return True

	# referring by others
	if mention[0].lower() in VALID_DETERMINERS and mention[-1].lower() in ["rest", "other", "others"]:
		return True

	return False

def find_str(s, char):
    index = 0
    indicies = []

    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index+len(char)] == char:
                    return index, index+len(char)
            index += 1

    return -1, -1

def output_brat_format(args, dialogue_corpus, batch_size=200):
	"""
		Source code for automatically detecting markables and outputting dialogues in brat format.
	"""
	global fixed_misspellings
	global replaced_strings
	global normalized_vocabs
	global inv_normalization_dict
	global split_morpheme
	global vocab

	# for detecting Noun Phrases
	corenlp_parser = CoreNLPParser(url='http://localhost:9000')

	# track mentions for manual analysis
	mention_counter = Counter()
	mention_context = defaultdict(list)
	context2mention = defaultdict(list)

	EOS_TOKENS = [".", "?", "!", "...", "!!!", "???"]

	batch_id = 0
	batch_len = 0

	for dialogue in tqdm(dialogue_corpus):
		batch_len += 1
		if batch_len % batch_size == 0:
			batch_id += 1

		if args.success_only and dialogue['outcome']['reward'] == 0:
			continue

		dialogue_txt = ""
		dialogue_ann = []
		total_chr_len = 0

		for event in dialogue["events"]:
			if event["action"] == "message":
				utterance_string = preprocess_utterance(args, event['data'])
				utterance_tokens = list(corenlp_parser.tokenize(utterance_string))

				aligned_string_tokens = []
				aligned_chr_len = 0
				for tok in utterance_tokens:
					if utterance_string[aligned_chr_len:].startswith(tok):
						aligned_string_tokens.append(tok)
						aligned_chr_len += len(tok)
					elif utterance_string[aligned_chr_len:].startswith(" "):
						aligned_string_tokens.append(" " + tok)
						aligned_chr_len += 1 + len(tok)
					else:
						pdb.set_trace()
				assert len(utterance_string) == sum([len(x) for x in aligned_string_tokens])
				assert len(utterance_tokens) == len(aligned_string_tokens)

				utterance_chr_len = len(utterance_string)

				# split utterances into sentences and parse NP
				current_sentence = []
				current_aligned_string = []
				mention_list = []

				# detokenize utterance
				current_string = "{}: ".format(event["agent"])
				current_chr_len = len(current_string)

				for i in range(len(utterance_tokens)):
					tok = utterance_tokens[i]
					aligned_string = aligned_string_tokens[i]
					if tok in EOS_TOKENS or i == len(utterance_tokens) - 1:
						current_sentence.append(tok)
						current_aligned_string.append(aligned_string)
						if len(current_sentence) > 0:
							sentence_string = "".join(current_aligned_string)
							const_tree = list(corenlp_parser.parse(current_sentence))
							mentions = detect_mentions(const_tree[0], unnested_only=True)
							unique_mentions = [list(x) for x in set(tuple(x) for x in mentions)]
							for mention in unique_mentions:
								if validate_mention(mention):
									with MosesDetokenizer('en') as detokenize:
										mention_string = detokenize(mention)
									mention_counter[mention_string] += 1
									mention_context[mention_string].append(current_sentence)
									mention_list.append(mention_string)
									for match in re.finditer(mention_string, sentence_string):
										start, end = match.span()
										# exclude hitting substrings, e.g. "it" in "with" 
										if sentence_string[start - 1] != ' ':
											continue
										if end < len(sentence_string) - 1 and sentence_string[end] not in ([" "] + PUNCTUATION):
											continue
										dialogue_ann.append((total_chr_len + current_chr_len + start, total_chr_len + current_chr_len + end, mention_string))
							if i == len(utterance_tokens) - 1:
								sentence_string += "\n"
							current_string += sentence_string
							current_chr_len += len(sentence_string)
							current_sentence = []
							current_aligned_string = []
					else:
						current_sentence.append(tok)
						current_aligned_string.append(aligned_string)

				assert len(current_string) == current_chr_len

				total_chr_len += current_chr_len
				dialogue_txt += current_string

				context2mention[utterance_string].append(list(set(mention_list)))

		# remove last \n
		dialogue_txt = dialogue_txt[:-1]
		total_chr_len -= 1

		if not os.path.exists("brat_format/batch_{:0>2}".format(batch_id)):
			os.makedirs("brat_format/batch_{:0>2}".format(batch_id))

		with open("brat_format/batch_{0:0>2}/{1}.txt".format(batch_id, dialogue["uuid"]), "w") as fout:
			fout.write(dialogue_txt)

		with open("brat_format/batch_{0:0>2}/{1}.ann".format(batch_id, dialogue["uuid"]), "w") as fout:
			for i, ann in enumerate(dialogue_ann):
				start, end, mention = ann
				fout.write("T{0}\tMarkable {1} {2}\t{3}\n".format(i+1, start, end, mention))

	for ctx, mentions in context2mention.items():
		print(ctx)
		print(mentions)


def output_markable_annotation(args, dialogue_corpus, fix_misspellings=False):
	"""
		Source code for converting brat annotated dialogues to json format.
		Markable ids will be given in the appearing order (starting from "M1").
	"""

	# all markables
	markables = defaultdict(dict)
	# markables in output format
	markable_annotation = {}

	# count markables in surface form
	markable_counter = Counter()
	# count markables in surface form
	markable2origin = defaultdict(dict)
	# count markable pos tag
	markable_pos = defaultdict(Counter)

	total_dialogues = 0

	finished_batches = [x for x in os.listdir('annotated') if x.startswith('batch_')]

	batch_info = defaultdict(list)
	chat_id2batch_id = {}

	for finished_batch in finished_batches:
		for filename in glob.glob('annotated/' + finished_batch + '/*.ann'):
			total_dialogues += 1
			chat_id = filename.split("/")[2].split(".")[0]
			batch_info[finished_batch].append(chat_id)
			chat_id2batch_id[chat_id] = finished_batch
			markable_annotation[chat_id] = {}

			with open(filename, "r") as fin:
				ann = fin.read()
				for line in ann.split("\n"):
					if line.split("\t")[0].startswith("T"):
						markable_id = line.split("\t")[0]
						label = line.split("\t")[1].split(" ")[0]
						text = line.split("\t")[-1]
						start = int(line.split("\t")[1].split(" ")[1])
						end = int(line.split("\t")[1].split(" ")[2])							
						markables[chat_id][markable_id] = Markable(markable_id, label, text, start, end)
					elif line.split("\t")[0].startswith("R"):
						arg1 = line.split("\t")[1].split(" ")[1].split(":")[1]
						arg2 = line.split("\t")[1].split(" ")[2].split(":")[1]
						if line.split("\t")[1].split(" ")[0] == "Anaphora":
							markables[chat_id][arg1].anaphora = arg2
						elif line.split("\t")[1].split(" ")[0] == "Cataphora":
							markables[chat_id][arg1].cataphora = arg2
						elif line.split("\t")[1].split(" ")[0] == "Predicative":
							markables[chat_id][arg1].predicative = arg2
					elif line.split("\t")[0].startswith("A"):
						arg = line.split("\t")[1].split(" ")[1]
						if line.split("\t")[1].split(" ")[0] == "No-Referent":
							markables[chat_id][arg].no_referent = True
						elif line.split("\t")[1].split(" ")[0] == "All-Referents":
							markables[chat_id][arg].all_referents = True
						elif line.split("\t")[1].split(" ")[0] == "Generic":
							markables[chat_id][arg].generic = True
					elif line.split("\t")[0].startswith("#"):
						# annotator note
						arg = line.split("\t")[1].split(" ")[1]
						annotator_note = line.split("\t")[2]
						#if editdistance.eval(markables[chat_id][arg].text, annotator_note) > 3:
						print(chat_id)
						print("\"{0}\" fixed to \"{1}\"".format(markables[chat_id][arg].text, annotator_note))
						# consider this as spelling correction
						markables[chat_id][arg].fixed_text = annotator_note								

	chat_ids = markables.keys()

	total_markables = 0
	annotating_markables = 0
	anaphora = 0
	cataphora = 0
	predicative = 0
	no_referent = 0
	all_referents = 0
	generic = 0
	for chat_id in chat_ids:
		with open('annotated/{0}/{1}.txt'.format(chat_id2batch_id[chat_id], chat_id), "r") as fin:
			text = fin.read()
		markable_annotation[chat_id]["text"] = text
		markable_annotation[chat_id]["markables"] = []

		fix_index = 0
		num_markables = 0
		fix_markable_idx = {}

		# sort markables in appearing order
		for original_id, markable in sorted(markables[chat_id].items(), key=lambda x: x[1].start):
			markable_dict = {}

			if markable.label == "None":
				if markable.fixed_text:
					markable_annotation[chat_id]["text"] = markable_annotation[chat_id]["text"][:markable.start + fix_index] + markable.fixed_text + markable_annotation[chat_id]["text"][markable.end + fix_index:]
					fix_index += len(markable.fixed_text) - len(markable.text)
			elif markable.label == "Markable":
				num_markables += 1
				fix_markable_idx[original_id] = "M{}".format(num_markables)
				markable_dict["markable_id"] = "M{}".format(num_markables)
				if markable.fixed_text:
					markable_dict["start"] = markable.start + fix_index
					markable_dict["end"] = markable.start + len(markable.fixed_text) + fix_index
					markable_annotation[chat_id]["text"] = markable_annotation[chat_id]["text"][:markable.start + fix_index] + markable.fixed_text + markable_annotation[chat_id]["text"][markable.end + fix_index:]
					markable_dict["text"] = markable.fixed_text
					fix_index += len(markable.fixed_text) - len(markable.text)
				else:
					markable_dict["start"] = markable.start + fix_index
					markable_dict["end"] = markable.end + fix_index
					markable_dict["text"] = markable.text

				markable_counter[markable_dict["text"]] += 1
				markable2origin[markable_dict["text"]]["chat_id"] = chat_id
				markable2origin[markable_dict["text"]]["batch_id"] = chat_id2batch_id[chat_id]					

				# temporal attribute of speaker information
				markable_dict["speaker"] = -1
				
				# attributes
				markable_dict["no-referent"] = markable.no_referent
				markable_dict["all-referents"] = markable.all_referents
				markable_dict["generic"] = markable.generic

				# relations (markable ids are fixed later)
				markable_dict["anaphora"] = markable.anaphora
				markable_dict["cataphora"] = markable.cataphora
				markable_dict["predicative"] = markable.predicative

				markable_annotation[chat_id]["markables"].append(markable_dict)
				assert markable_annotation[chat_id]["text"][markable_dict["start"]:markable_dict["end"]] == markable_dict["text"]

				# save basic statistics of markables
				total_markables += 1
				if markable.anaphora is not None:
					anaphora += 1
				elif markable.cataphora is not None:
					cataphora += 1
				elif markable.predicative is not None:
					predicative += 1
				elif markable.no_referent:
					no_referent += 1
				elif markable.all_referents:
					all_referents += 1
				elif markable.generic:
					generic += 1
				else:
					annotating_markables += 1

				if markable.predicative is None and not markable.generic:
					for tok, pos in pos_tag(word_tokenize(markable_dict["text"])):
						markable_pos[pos][tok.lower()] += 1
			else:
				pdb.set_trace()
		
		# fix markable ids for relations
		for markable_dict in markable_annotation[chat_id]["markables"]:
			if markable_dict["anaphora"]:
				markable_dict["anaphora"] = fix_markable_idx[markable_dict["anaphora"]]
			if markable_dict["cataphora"]:
				markable_dict["cataphora"] = fix_markable_idx[markable_dict["cataphora"]]
			if markable_dict["predicative"]:
				markable_dict["predicative"] = fix_markable_idx[markable_dict["predicative"]]

	# add speaker information
	for chat_id in markable_annotation.keys():
		total_len = 0
		for line in markable_annotation[chat_id]["text"].split("\n"):
			if line.startswith("0:"):
				speaker = 0
			elif line.startswith("1:"):
				speaker = 1
			else:
				assert False
			total_len += len(line) + 1 # add len of "\n"
			for markable_dict in markable_annotation[chat_id]["markables"]:
				if markable_dict["speaker"] < 0 and markable_dict["start"] <= total_len:
					markable_dict["speaker"] = speaker
				if markable_dict["start"] > total_len:
					break

	print("total dialogues: {}".format(total_dialogues))
	print("total markables: {}".format(total_markables - predicative - generic))
	print("annotating markables: {}".format(annotating_markables))
	print("no_referent: {}".format(no_referent))
	print("all_referents: {}".format(all_referents))
	print("generic: {}".format(generic))
	print("anaphora: {}".format(anaphora))
	print("cataphora: {}".format(cataphora))
	print("predicative: {}".format(predicative))

	dump_json(markable_annotation, "markable_annotation.json")

	dump_json(batch_info, "batch_info.json")

	start_word = sorted(set([x.split(" ")[0].lower() for x in markable_counter]))
	end_word = sorted(set([x.split(" ")[-1].lower() for x in markable_counter]))

	[x for x in markable_counter if x.split(" ")[0] == ""]

	total_tokens = sum([sum(markable_pos[pos].values()) for pos in markable_pos.keys()])
	for pos in markable_pos.keys():
		if sum(markable_pos[pos].values()) > 100:
			print("{}: unique {} | total {} ({:.2f}%)".format(pos, len(markable_pos[pos].keys()), sum(markable_pos[pos].values()), 100.0 * sum(markable_pos[pos].values()) / total_tokens))

def markable_agreement(args, dialogue_corpus, annotators=[], include_errors=False):
	"""
		Compute the agreement of markable detections.
		Agreements are computed at the token level for simplicity.
	"""

	def _compute_agreement(chat_ids, candidate_tokens, valid_annotators, annotation_idx):
		total_pairwise_judgements = 0
		total_is_markable = 0
		total_is_not_markable = 0			
		num_pairwise_agreement = 0

		for chat_id in chat_ids:
			total_pairwise_judgements += len(candidate_tokens[chat_id]["tokens"]) * (len(valid_annotators) * (len(valid_annotators) - 1) / 2)

			for idx in range(len(candidate_tokens[chat_id]["tokens"])):
				is_markable = 0
				is_not_markable = 0
				for annotator in valid_annotators:
					if idx in annotation_idx[annotator][chat_id]:
						is_markable += 1
					else:
						is_not_markable += 1
				num_pairwise_agreement += is_markable * (is_markable - 1) / 2
				num_pairwise_agreement += is_not_markable * (is_not_markable - 1) / 2
				total_is_markable += is_markable
				total_is_not_markable += is_not_markable

		observed_agreement = 1.0 * num_pairwise_agreement / total_pairwise_judgements

		# fleiss's multi-pi
		probability_is_markable = total_is_markable / (total_is_markable + total_is_not_markable)

		expected_agreement = probability_is_markable * probability_is_markable + (1 - probability_is_markable) * (1 - probability_is_markable)

		multi_pi = (observed_agreement - expected_agreement) / (1 - expected_agreement)

		print("total: {}".format(total_is_markable))
		print("observed_agreement: {}".format(observed_agreement))
		print("fleiss's multi-pi: {}".format(multi_pi))

	markables = {}
	candidate_tokens = {}
	valid_annotators = []
	chat_ids = set()

	for annotator in annotators:
		if os.path.exists('annotated/' + str(annotator)):
			markables[annotator] = {}
			valid_annotators.append(annotator)
		else:
			continue

		for filename in glob.glob('annotated/' + str(annotator) + '/' + args.batch_id + '/*.ann'):
			chat_id = filename.split("/")[3].split(".")[0]
			chat_ids.add(chat_id)

			markables[annotator][chat_id] = {}

			# read text to calculate candidate_tokens
			if not chat_id in candidate_tokens:
				candidate_tokens[chat_id] = {}
				candidate_tokens[chat_id]["tokens"] = []
				candidate_tokens[chat_id]["start_idx"] = []
				with open('annotated/' + str(annotator) + '/' + args.batch_id + '/{}.txt'.format(chat_id), "r") as fin:
					text = fin.read()
				start_idx = 0
				for line in text.split("\n"):
					for tok in line.split(" "): # skip speaker info
						candidate_tokens[chat_id]["tokens"].append(tok)
						candidate_tokens[chat_id]["start_idx"].append(start_idx)
						start_idx += len(tok) + 1
				assert len(text) == len(" ".join(candidate_tokens[chat_id]["tokens"]))

			with open(filename, "r") as fin:
				ann = fin.read()
				for line in ann.split("\n"):
					if line.split("\t")[0].startswith("T"):
						markable_id = line.split("\t")[0]
						label = line.split("\t")[1].split(" ")[0]
						text = line.split("\t")[-1]
						start = int(line.split("\t")[1].split(" ")[1])
						end = int(line.split("\t")[1].split(" ")[2])
						if label == "Markable": # ignore "None"
							markables[annotator][chat_id][markable_id] = Markable(markable_id, label, text, start, end)
					elif line.split("\t")[0].startswith("R"):
						arg1 = line.split("\t")[1].split(" ")[1].split(":")[1]
						arg2 = line.split("\t")[1].split(" ")[2].split(":")[1]
						if line.split("\t")[1].split(" ")[0] == "Anaphora":
							markables[annotator][chat_id][arg1].anaphora = arg2
						elif line.split("\t")[1].split(" ")[0] == "Cataphora":
							markables[annotator][chat_id][arg1].cataphora = arg2
						elif line.split("\t")[1].split(" ")[0] == "Predicative":
							markables[annotator][chat_id][arg1].predicative = arg2
					elif line.split("\t")[0].startswith("A"):
						arg = line.split("\t")[1].split(" ")[1]
						if line.split("\t")[1].split(" ")[0] == "No-Referent":
							markables[annotator][chat_id][arg].no_referent = True
						elif line.split("\t")[1].split(" ")[0] == "All-Referents":
							markables[annotator][chat_id][arg].all_referents = True
						elif line.split("\t")[1].split(" ")[0] == "Generic":
							markables[annotator][chat_id][arg].generic = True
			markables[annotator][chat_id] = sorted(markables[annotator][chat_id].items(), key=lambda x: x[1].start)

	chat_ids = list(chat_ids)

	# compute start/end idx (in tokens) of annotated markables
	markable_start_idx = {}
	markable_end_idx = {}
	all_referents_idx = {}
	no_referent_idx = {}
	anaphora_pairs = {}
	cataphora_pairs = {}
	for annotator in valid_annotators:
		markable_start_idx[annotator] = {}
		markable_end_idx[annotator] = {}
		all_referents_idx[annotator] = {}
		no_referent_idx[annotator] = {}
		anaphora_pairs[annotator] = {}
		cataphora_pairs[annotator] = {}
		for chat_id in chat_ids:
			# compute start idx
			markable_start_idx[annotator][chat_id] = []
			for markable_id, markable in markables[annotator][chat_id]:
				# these are not counted as markables
				if markable.generic or markable.predicative:
					continue
				idx = 0
				for i in range(len(candidate_tokens[chat_id]["start_idx"])):
					if i == len(candidate_tokens[chat_id]["start_idx"]) - 1:
						idx = i
					elif candidate_tokens[chat_id]["start_idx"][i] - 1 <= markable.start and markable.start < candidate_tokens[chat_id]["start_idx"][i+1] - 1:
						idx = i
						break
				markable_start_idx[annotator][chat_id].append(idx)

			# compute end idx
			markable_end_idx[annotator][chat_id] = []
			for markable_id, markable in markables[annotator][chat_id]:
				# these are not counted as markables
				if markable.generic or markable.predicative:
					continue
				idx = 0
				for i in range(len(candidate_tokens[chat_id]["start_idx"])):
					if i == len(candidate_tokens[chat_id]["start_idx"]) - 1:
						idx = i
					elif candidate_tokens[chat_id]["start_idx"][i] < markable.end and markable.end <= candidate_tokens[chat_id]["start_idx"][i+1]:
						idx = i
						break
				markable_end_idx[annotator][chat_id].append(idx)

			# compute all referents idx
			all_referents_idx[annotator][chat_id] = []
			for markable_id, markable in markables[annotator][chat_id]:
				# these are not counted as markables
				if markable.generic or markable.predicative:
					continue
				if markable.all_referents:
					idx = 0
					for i in range(len(candidate_tokens[chat_id]["start_idx"])):
						if i == len(candidate_tokens[chat_id]["start_idx"]) - 1:
							idx = i
						elif candidate_tokens[chat_id]["start_idx"][i] < markable.end and markable.end <= candidate_tokens[chat_id]["start_idx"][i+1]:
							idx = i
							break
					all_referents_idx[annotator][chat_id].append(idx)

			# compute no referent idx
			no_referent_idx[annotator][chat_id] = []
			for markable_id, markable in markables[annotator][chat_id]:
				# these are not counted as markables
				if markable.generic or markable.predicative:
					continue
				if markable.no_referent:
					idx = 0
					for i in range(len(candidate_tokens[chat_id]["start_idx"])):
						if i == len(candidate_tokens[chat_id]["start_idx"]) - 1:
							idx = i
						elif candidate_tokens[chat_id]["start_idx"][i] < markable.end and markable.end <= candidate_tokens[chat_id]["start_idx"][i+1]:
							idx = i
							break
					no_referent_idx[annotator][chat_id].append(idx)

			# compute anaphora pairs
			anaphora_pairs[annotator][chat_id] = []
			for markable_id, markable in markables[annotator][chat_id]:
				# these are not counted as markables
				if markable.generic or markable.predicative:
					continue
				if markable.anaphora:
					anaphora_idx = 0
					for i in range(len(candidate_tokens[chat_id]["start_idx"])):
						if i == len(candidate_tokens[chat_id]["start_idx"]) - 1:
							anaphora_idx = i
						elif candidate_tokens[chat_id]["start_idx"][i] < markable.end and markable.end <= candidate_tokens[chat_id]["start_idx"][i+1]:
							anaphora_idx = i
							break
					
					for _markable in markables[annotator][chat_id]:
						if _markable[0] == markable.anaphora:
							antecedent_markable = _markable[1]
							antecedent_idx = 0
							for i in range(len(candidate_tokens[chat_id]["start_idx"])):
								if i == len(candidate_tokens[chat_id]["start_idx"]) - 1:
									antecedent_idx = i
								elif candidate_tokens[chat_id]["start_idx"][i] < antecedent_markable.end and antecedent_markable.end <= candidate_tokens[chat_id]["start_idx"][i+1]:
									antecedent_idx = i
									break
							anaphora_pairs[annotator][chat_id].append((antecedent_idx, anaphora_idx))

			# compute cataphora pairs
			cataphora_pairs[annotator][chat_id] = []
			for markable_id, markable in markables[annotator][chat_id]:
				# these are not counted as markables
				if markable.generic or markable.predicative:
					continue
				if markable.cataphora:
					cataphora_idx = 0
					for i in range(len(candidate_tokens[chat_id]["start_idx"])):
						if i == len(candidate_tokens[chat_id]["start_idx"]) - 1:
							cataphora_idx = i
						elif candidate_tokens[chat_id]["start_idx"][i] < markable.end and markable.end <= candidate_tokens[chat_id]["start_idx"][i+1]:
							cataphora_idx = i
							break
					
					for _markable in markables[annotator][chat_id]:
						if _markable[0] == markable.cataphora:
							decedent_markable = _markable[1]
							decedent_idx = 0
							for i in range(len(candidate_tokens[chat_id]["start_idx"])):
								if i == len(candidate_tokens[chat_id]["start_idx"]) - 1:
									decedent_idx = i
								elif candidate_tokens[chat_id]["start_idx"][i] < decedent_markable.end and decedent_markable.end <= candidate_tokens[chat_id]["start_idx"][i+1]:
									decedent_idx = i
									break
							cataphora_pairs[annotator][chat_id].append((cataphora_idx, decedent_idx))

	if args.fix_agreement:
		print("fix agreement")
		removed = 0
		for annotator in valid_annotators:
			for chat_id in chat_ids:
				for i in range(1, len(candidate_tokens[chat_id]["tokens"]) - 1):
					tok = candidate_tokens[chat_id]["tokens"][i]
					if tok == "of" and i - 1 in markable_end_idx[annotator][chat_id] and i + 1 in markable_start_idx[annotator][chat_id]:
						markable_start_idx[annotator][chat_id].remove(i + 1)
						removed += 1
		print("removed markables: {}".format(removed))		

	print("start agreement:")
	_compute_agreement(chat_ids, candidate_tokens, valid_annotators, markable_start_idx)
	print("")

	print("end agreement:")
	_compute_agreement(chat_ids, candidate_tokens, valid_annotators, markable_end_idx)
	print("")

	print("all referents agreement:")
	_compute_agreement(chat_ids, candidate_tokens, valid_annotators, all_referents_idx)
	print("")

	print("no referent agreement:")
	_compute_agreement(chat_ids, candidate_tokens, valid_annotators, no_referent_idx)
	print("")

	print("anaphora agreement:")
	candidate_pairs = {}
	for chat_id in chat_ids:
		# compute candidate pairs
		candidate_pairs[chat_id] = []
		for i in range(len(candidate_tokens[chat_id]["tokens"])):
			for j in range(i + 1, len(candidate_tokens[chat_id]["tokens"])):
				candidate_pairs[chat_id].append((i,j))
				if j == len(candidate_tokens[chat_id]["tokens"]) or candidate_tokens[chat_id]["tokens"][j] in ["0:", "1:"]:
					break

	total_pairwise_judgements = 0
	total_is_anaphora_pair = 0
	total_is_not_anaphora_pair = 0			
	num_pairwise_agreement = 0
	for chat_id in chat_ids:
		total_pairwise_judgements += len(candidate_pairs[chat_id]) * (len(valid_annotators) * (len(valid_annotators) - 1) / 2)

		for idx_pair in candidate_pairs[chat_id]:
			is_anaphora_pair = 0
			is_not_anaphora_pair = 0
			for annotator in valid_annotators:
				if idx_pair in anaphora_pairs[annotator][chat_id]:
					is_anaphora_pair += 1
				else:
					is_not_anaphora_pair += 1
			num_pairwise_agreement += is_anaphora_pair * (is_anaphora_pair - 1) / 2
			num_pairwise_agreement += is_not_anaphora_pair * (is_not_anaphora_pair - 1) / 2
			total_is_anaphora_pair += is_anaphora_pair
			total_is_not_anaphora_pair += is_not_anaphora_pair

	observed_agreement = 1.0 * num_pairwise_agreement / total_pairwise_judgements

	# fleiss's multi-pi
	probability_is_anaphora_pair = total_is_anaphora_pair / (total_is_anaphora_pair + total_is_not_anaphora_pair)

	expected_agreement = probability_is_anaphora_pair * probability_is_anaphora_pair + (1 - probability_is_anaphora_pair) * (1 - probability_is_anaphora_pair)

	multi_pi = (observed_agreement - expected_agreement) / (1 - expected_agreement)

	print("total: {}".format(total_is_anaphora_pair))
	print("observed_agreement: {}".format(observed_agreement))
	print("fleiss's multi-pi: {}".format(multi_pi))
	print("")

	print("cataphora agreement:")
	total_pairwise_judgements = 0
	total_is_pair = 0
	total_is_not_pair = 0			
	num_pairwise_agreement = 0
	for chat_id in chat_ids:
		total_pairwise_judgements += len(candidate_pairs[chat_id]) * (len(valid_annotators) * (len(valid_annotators) - 1) / 2)

		for idx_pair in candidate_pairs[chat_id]:
			is_pair = 0
			is_not_pair = 0
			for annotator in valid_annotators:
				if idx_pair in cataphora_pairs[annotator][chat_id]:
					is_pair += 1
				else:
					is_not_pair += 1
			num_pairwise_agreement += is_pair * (is_pair - 1) / 2
			num_pairwise_agreement += is_not_pair * (is_not_pair - 1) / 2
			total_is_pair += is_pair
			total_is_not_pair += is_not_pair

	observed_agreement = 1.0 * num_pairwise_agreement / total_pairwise_judgements

	# fleiss's multi-pi
	probability_is_pair = total_is_pair / (total_is_pair + total_is_not_pair)

	expected_agreement = probability_is_pair * probability_is_pair + (1 - probability_is_pair) * (1 - probability_is_pair)

	multi_pi = (observed_agreement - expected_agreement) / (1 - expected_agreement)

	print("total: {}".format(total_is_pair))
	print("observed_agreement: {}".format(observed_agreement))
	print("fleiss's multi-pi: {}".format(multi_pi))
	print("")

def referent_agreement(args, dialogue_corpus):
	"""
		Compute the agreements of referent resolution based on exact-match and entity-level judgements.
	"""
	markable_annotation = read_json("markable_annotation.json")
	referent_annotation = read_json("referent_annotation.json")
	aggregated_referent_annotation = read_json("aggregated_referent_annotation.json")
	chat_ids = list(referent_annotation.keys())

	""" First, compute pairwise agreement of crowdworkers """
	print("=" * 7)
	print("computing pairwise agreement of crowdworkers")
	print("=" * 7)

	total_pairwise_entlevel_judgements = 0
	total_pairwise_judgements = 0
	total_is_referent = 0
	total_is_not_referent = 0			
	num_pairwise_entlevel_agreement = 0
	total_markables = 0
	total_exact_match = 0
	total_judgements = 0
	num_pairwise_exact_match = 0
	num_referents_counter = Counter()
	exact_match_counter = Counter()

	exact_match_rate = {}

	for dialogue in dialogue_corpus:
		chat_id = dialogue["uuid"]
		if not chat_id in markable_annotation:
			continue
		else:
			markables = markable_annotation[chat_id]
		if not chat_id in referent_annotation:
			continue

		mturk_workers = [x for x in list(referent_annotation[chat_id].keys()) if x.startswith("MT_")]
		if len(mturk_workers) < 3:
			continue

		agent_0_kb, agent_1_kb = dialogue["scenario"]["kbs"]
		agent_0_ents = ["agent_0_{}".format(x['id']) for x in agent_0_kb]
		agent_1_ents = ["agent_1_{}".format(x['id']) for x in agent_1_kb]

		num_judgement = 0
		num_exact_match = 0
		for markable in markables["markables"]:
			markable_id = markable["markable_id"]
			speaker = markable["speaker"]
			candidate_ents = agent_0_ents if speaker == 0 else agent_1_ents

			if is_annotatable_markable(markable):
				total_markables += 1

				for ent in candidate_ents:
					is_referent = 0
					is_not_referent = 0
					for annotator in mturk_workers:
						if ent in referent_annotation[chat_id][annotator][markable_id]["referents"]:
							is_referent += 1
						else:
							is_not_referent += 1
					num_pairwise_entlevel_agreement += is_referent * (is_referent - 1) / 2
					num_pairwise_entlevel_agreement += is_not_referent * (is_not_referent - 1) / 2
					total_is_referent += is_referent
					total_is_not_referent += is_not_referent

				total_pairwise_entlevel_judgements += (len(mturk_workers) * (len(mturk_workers) - 1) / 2) * len(candidate_ents)
				total_pairwise_judgements += (len(mturk_workers) * (len(mturk_workers) - 1) / 2)
				total_judgements += len(mturk_workers)

				exact_match = True
				for a, b in itertools.combinations(mturk_workers, 2):
					a_set = set(referent_annotation[chat_id][a][markable_id]['referents'])
					b_set = set(referent_annotation[chat_id][b][markable_id]['referents'])
					if a_set == b_set:
						exact_match_counter[len(a_set)] += 1
						exact_match_counter[len(b_set)] += 1
						num_pairwise_exact_match += 1
					else:
						exact_match = False
					num_referents_counter[len(a_set)] += 1
					num_referents_counter[len(b_set)] += 1

				if exact_match:
					total_exact_match += 1
					num_exact_match += 1
				num_judgement += 1
		exact_match_rate[chat_id] = 1.0 * num_exact_match / num_judgement

	observed_agreement = 1.0 * num_pairwise_entlevel_agreement / total_pairwise_entlevel_judgements

	# fleiss's multi-pi
	probability_is_referent = total_is_referent / (total_is_referent + total_is_not_referent)

	expected_agreement = probability_is_referent * probability_is_referent + (1 - probability_is_referent) * (1 - probability_is_referent)

	multi_pi = (observed_agreement - expected_agreement) / (1 - expected_agreement)

	print("observed_agreement: {}".format(observed_agreement))
	print("fleiss's multi-pi: {}".format(multi_pi))
	print("total markables: {}".format(total_markables))
	print("exact match: {}".format(1.0 * num_pairwise_exact_match / total_pairwise_judgements))
	print("total judgements: {}".format(total_judgements))
	print("total dialogues: {}".format(len(chat_ids)))

	for num_referents in num_referents_counter.keys():
		print('{}: {:.4f} (% judgements: {})'.format(num_referents, exact_match_counter[num_referents] / num_referents_counter[num_referents], 100.0 * num_referents_counter[num_referents] / sum(num_referents_counter.values())))

	exact_match_rate = list(exact_match_rate.values())
	sns.distplot(exact_match_rate, kde=False, rug=False)
	plt.savefig('exact_match_rate.png', dpi=300)

	""" Next, compute agreement with the admin"""
	print("=" * 7)
	print("computing agreement with admin")
	print("=" * 7)

	total_pairwise_judgements = 0
	total_is_referent = 0
	total_is_not_referent = 0			
	num_pairwise_agreement = 0
	total_markables = 0
	total_exact_match = 0
	total_judgements = 0
	num_referents_counter = Counter()
	exact_match_counter = Counter()

	exact_match_rate = {}

	admin_annotated_chat_ids = []
	num_aggregated_agreement = 0
	num_aggregated_judgement = 0
	total_aggregated_exact_match = 0
	total_aggregated_judgements = 0
	aggregated_num_referents_counter = Counter()
	aggregated_exact_match_counter = Counter()

	for dialogue in dialogue_corpus:
		chat_id = dialogue["uuid"]
		if not chat_id in markable_annotation:
			continue
		else:
			markables = markable_annotation[chat_id]
		if not chat_id in referent_annotation:
			continue
		if not "admin" in referent_annotation[chat_id]:
			continue
		else:
			admin_annotated_chat_ids.append(chat_id)

		mturk_workers = [x for x in list(referent_annotation[chat_id].keys()) if x.startswith("MT_")]
		if len(mturk_workers) < 3:
			continue

		agent_0_kb, agent_1_kb = dialogue["scenario"]["kbs"]
		agent_0_ents = ["agent_0_{}".format(x['id']) for x in agent_0_kb]
		agent_1_ents = ["agent_1_{}".format(x['id']) for x in agent_1_kb]

		num_judgement = 0
		num_exact_match = 0
		for markable in markables["markables"]:
			markable_id = markable["markable_id"]
			speaker = markable["speaker"]
			candidate_ents = agent_0_ents if speaker == 0 else agent_1_ents

			if is_annotatable_markable(markable):
				#if aggregated_referent_annotation[chat_id][markable_id]["unidentifiable"]:
				#	continue

				total_markables += 1

				is_referent = 0
				is_not_referent = 0

				if ent in referent_annotation[chat_id]["admin"][markable_id]["referents"]:
					is_referent += 1
				else:
					is_not_referent += 1

				for annotator in mturk_workers:
					for ent in candidate_ents:
						if ent in referent_annotation[chat_id][annotator][markable_id]["referents"]:
							is_referent += 1
							if ent in referent_annotation[chat_id]["admin"][markable_id]["referents"]:
								num_pairwise_agreement += 1
						else:
							is_not_referent += 1
							if ent not in referent_annotation[chat_id]["admin"][markable_id]["referents"]:
								num_pairwise_agreement += 1

				for ent in candidate_ents:
					if ent in aggregated_referent_annotation[chat_id][markable_id]["referents"]:
						if ent in referent_annotation[chat_id]["admin"][markable_id]["referents"]:
							num_aggregated_agreement += 1
					else:
						is_not_referent += 1
						if ent not in referent_annotation[chat_id]["admin"][markable_id]["referents"]:
							num_aggregated_agreement += 1
					num_aggregated_judgement += 1

				total_is_referent += is_referent
				total_is_not_referent += is_not_referent

				total_pairwise_judgements += len(mturk_workers) * 7

				exact_match = True

				for annotator in mturk_workers:
					admin_set = set(referent_annotation[chat_id]["admin"][markable_id]['referents'])
					annotator_set = set(referent_annotation[chat_id][annotator][markable_id]['referents'])
					if admin_set == annotator_set:
						exact_match_counter[len(admin_set)] += 1
						total_exact_match += 1
					num_referents_counter[len(admin_set)] += 1
					total_judgements += 1

				admin_set = set(referent_annotation[chat_id]["admin"][markable_id]['referents'])
				aggregated_set = set(aggregated_referent_annotation[chat_id][markable_id]['referents'])	
				if admin_set == aggregated_set:
					aggregated_exact_match_counter[len(admin_set)] += 1
					total_aggregated_exact_match += 1
				aggregated_num_referents_counter[len(admin_set)] += 1
				total_aggregated_judgements += 1			

	observed_agreement = 1.0 * num_pairwise_agreement / total_pairwise_judgements

	# fleiss's multi-pi
	probability_is_referent = total_is_referent / (total_is_referent + total_is_not_referent)

	expected_agreement = probability_is_referent * probability_is_referent + (1 - probability_is_referent) * (1 - probability_is_referent)

	multi_pi = (observed_agreement - expected_agreement) / (1 - expected_agreement)

	print("observed_agreement: {}".format(observed_agreement))
	print("fleiss's multi-pi: {}".format(multi_pi))
	print("total markables: {}".format(total_markables))
	print("exact match: {}".format(1.0 * total_exact_match / total_judgements))
	print("total judgements: {}".format(total_judgements))
	print("total dialogues: {}".format(len(admin_annotated_chat_ids)))

	for num_referents in num_referents_counter.keys():
		print('{}: {:.4f} (% judgements: {})'.format(num_referents, exact_match_counter[num_referents] / num_referents_counter[num_referents], 100.0 * num_referents_counter[num_referents] / sum(num_referents_counter.values())))

	exact_match_rate = list(exact_match_rate.values())
	sns.distplot(exact_match_rate, kde=False, rug=False)
	plt.savefig('admin_exact_match_rate.png', dpi=300)

	print("=" * 7)
	print("computing agreement with admin and aggregated")
	print("=" * 7)

	observed_agreement = 1.0 * num_aggregated_agreement / num_aggregated_judgement

	# fleiss's multi-pi
	probability_is_referent = total_is_referent / (total_is_referent + total_is_not_referent)

	expected_agreement = probability_is_referent * probability_is_referent + (1 - probability_is_referent) * (1 - probability_is_referent)

	multi_pi = (observed_agreement - expected_agreement) / (1 - expected_agreement)

	print("observed_agreement: {}".format(observed_agreement))
	print("fleiss's multi-pi: {}".format(multi_pi))
	print("total markables: {}".format(total_markables))
	print("exact match: {}".format(1.0 * total_aggregated_exact_match / total_aggregated_judgements))
	print("total judgements: {}".format(total_aggregated_judgements))
	print("total dialogues: {}".format(len(admin_annotated_chat_ids)))

	for num_referents in aggregated_num_referents_counter.keys():
		print('{}: {:.4f} (% judgements: {})'.format(num_referents, aggregated_exact_match_counter[num_referents] / aggregated_num_referents_counter[num_referents], 100.0 * aggregated_num_referents_counter[num_referents] / sum(aggregated_num_referents_counter.values())))

def referent_aggregation(args, dialogue_corpus):
	markable_annotation = read_json("markable_annotation.json")
	referent_annotation = read_json("referent_annotation.json")
	chat_ids = list(referent_annotation.keys())

	aggregated_referent_annotation = {}

	num_referents_counter = Counter()

	ambiguous_judgements = 0
	unidentifiable_judgements = 0
	total_judgements = 0

	for dialogue in dialogue_corpus:
		chat_id = dialogue["uuid"]
		if not chat_id in markable_annotation:
			continue
		else:
			markables = markable_annotation[chat_id]
		if not chat_id in referent_annotation:
			continue

		mturk_workers = [x for x in list(referent_annotation[chat_id].keys()) if x.startswith("MT_")]
		if len(mturk_workers) < 3:
			continue
		else:
			aggregated_referent_annotation[chat_id] = {}

		agent_0_kb, agent_1_kb = dialogue["scenario"]["kbs"]
		agent_0_ents = ["agent_0_{}".format(x['id']) for x in agent_0_kb]
		agent_1_ents = ["agent_1_{}".format(x['id']) for x in agent_1_kb]

		for markable in markables["markables"]:
			markable_id = markable["markable_id"]
			speaker = markable["speaker"]
			candidate_ents = agent_0_ents if speaker == 0 else agent_1_ents

			if is_annotatable_markable(markable):
				aggregated_referent_annotation[chat_id][markable_id] = {}
				aggregated_referent_annotation[chat_id][markable_id]["referents"] = []

				for ent in candidate_ents:
					is_referent = 0
					is_not_referent = 0
					for annotator in mturk_workers:
						if ent in referent_annotation[chat_id][annotator][markable_id]["referents"]:
							is_referent += 1
						else:
							is_not_referent += 1
					if is_referent >= len(mturk_workers) / 2:
						aggregated_referent_annotation[chat_id][markable_id]["referents"].append(ent)

				is_ambiguous = 0
				is_unidentifiable = 0
				for annotator in mturk_workers:
					num_referents_counter[len(referent_annotation[chat_id][annotator][markable_id]["referents"])] += 1
					if referent_annotation[chat_id][annotator][markable_id]["ambiguous"]:
						is_ambiguous += 1
						ambiguous_judgements += 1
					if referent_annotation[chat_id][annotator][markable_id]["unidentifiable"]:
						is_unidentifiable += 1
						unidentifiable_judgements += 1
					total_judgements += 1
				if is_ambiguous >= len(mturk_workers) / 2:
					aggregated_referent_annotation[chat_id][markable_id]["ambiguous"] = True
				else:
					aggregated_referent_annotation[chat_id][markable_id]["ambiguous"] = False
				if is_unidentifiable >= len(mturk_workers) / 2:
					aggregated_referent_annotation[chat_id][markable_id]["unidentifiable"] = True
				else:
					aggregated_referent_annotation[chat_id][markable_id]["unidentifiable"] = False

	total = 0
	ambiguous = 0
	unidentifiable = 0
	ambiguous_counter = Counter()
	unidentifiable_counter = Counter()
	num_referents = Counter()
	num_referred_per_dialogue = Counter()
	automatically_annotated = 0

	for chat_id in list(aggregated_referent_annotation.keys()):
		chat = [chat for chat in dialogue_corpus if chat['uuid'] == chat_id]
		chat = chat[0]
		agent_0_kb, agent_1_kb = chat["scenario"]["kbs"]
		agent_0_ents = ["agent_0_{}".format(x['id']) for x in agent_0_kb]
		agent_1_ents = ["agent_1_{}".format(x['id']) for x in agent_1_kb]

		agent_0_referenced = set()
		agent_1_referenced = set()
		for markable_id in list(aggregated_referent_annotation[chat_id].keys()):
			total += 1
			num_referents[len(aggregated_referent_annotation[chat_id][markable_id]["referents"])] += 1
			if aggregated_referent_annotation[chat_id][markable_id]["ambiguous"]:
				ambiguous += 1
				ambiguous_counter[len(aggregated_referent_annotation[chat_id][markable_id]["referents"])] += 1
			if aggregated_referent_annotation[chat_id][markable_id]["unidentifiable"]:
				unidentifiable += 1
				unidentifiable_counter[len(aggregated_referent_annotation[chat_id][markable_id]["referents"])] += 1
			for referent in aggregated_referent_annotation[chat_id][markable_id]["referents"]:
				if referent.startswith("agent_0"):
					agent_0_referenced.add(referent)
				else:
					agent_1_referenced.add(referent)
		num_referred_per_dialogue[len(agent_0_referenced)] += 1
		num_referred_per_dialogue[len(agent_1_referenced)] += 1

		# automatically annotate markables:
		markables = markable_annotation[chat_id]
		#while len(aggregated_referent_annotation[chat_id].keys()) < len(markables["markables"])
		while True:
			modified = False
			for markable in markables["markables"]:
				markable_id = markable["markable_id"]
				speaker = markable["speaker"]
				if not markable_id in aggregated_referent_annotation[chat_id]:
					if markable["anaphora"]:
						anaphora_id = markable["anaphora"]
						if anaphora_id in aggregated_referent_annotation[chat_id]:
							aggregated_referent_annotation[chat_id][markable_id] = aggregated_referent_annotation[chat_id][anaphora_id]
							modified = True
							automatically_annotated += 1
					elif markable["cataphora"]:
						cataphora_id = markable["cataphora"]
						if cataphora_id in aggregated_referent_annotation[chat_id]:
							aggregated_referent_annotation[chat_id][markable_id] = aggregated_referent_annotation[chat_id][cataphora_id]
							modified = True
							automatically_annotated += 1
					elif markable["all-referents"]:
						if markable["speaker"] == 0:
							aggregated_referent_annotation[chat_id][markable_id] = {}
							aggregated_referent_annotation[chat_id][markable_id]["referents"] = agent_0_ents
							modified = True
							automatically_annotated += 1
						else:
							aggregated_referent_annotation[chat_id][markable_id] = {}
							aggregated_referent_annotation[chat_id][markable_id]["referents"] = agent_1_ents
							modified = True
							automatically_annotated += 1
					elif markable["no-referent"]:
						aggregated_referent_annotation[chat_id][markable_id] = {}
						aggregated_referent_annotation[chat_id][markable_id]["referents"] = []
						modified = True
						automatically_annotated += 1
			if not modified:
				break

	print("manually annotated: {}".format(total))
	print("ambiguous judgements: {:.5f}".format(100.0 * ambiguous_judgements / total_judgements))
	print("ambiguous (aggregated): {:.5f}".format(1.0 * ambiguous / total))
	print("unidentifiable judgements: {}".format(100.0 * unidentifiable_judgements / total_judgements))
	print("unidentifiable (aggregated): {}".format(1.0 * unidentifiable / total))
	print("num referents per markable: {}".format(num_referents))
	print("num unique referents per dialogue: {}".format(num_referred_per_dialogue))
	print("automatically annotated: {}".format(automatically_annotated))

	for num_referents in num_referents_counter.keys():
		print('{}: {}'.format(num_referents, 100.0 * num_referents_counter[num_referents] / sum(num_referents_counter.values())))

	dump_json(aggregated_referent_annotation, "aggregated_referent_annotation.json")

def referent_color(args, dialogue_corpus):
	markable_annotation = read_json("markable_annotation.json")
	referent_annotation = read_json("aggregated_referent_annotation.json")
	chat_ids = list(referent_annotation.keys())

	text2color = defaultdict(list)

	for dialogue in dialogue_corpus:
		chat_id = dialogue["uuid"]
		if not chat_id in markable_annotation:
			continue
		else:
			markables = markable_annotation[chat_id]
		if not chat_id in referent_annotation:
			continue

		agent_0_kb, agent_1_kb = dialogue["scenario"]["kbs"]
		ent_id2color = {x['id'] : int(x['color'].split(',')[1]) for x in agent_0_kb + agent_1_kb}
		#agent_0_ents = ["agent_0_{}".format(x['id']) for x in agent_0_kb]
		#agent_0_colors = [int(x['color'].split(',')[1]) for x in agent_0_kb]
		#agent_1_ents = ["agent_1_{}".format(x['id']) for x in agent_1_kb]
		#agent_1_colors = [int(x['color'].split(',')[1]) for x in agent_1_kb]

		for markable in markables["markables"]:
			markable_id = markable["markable_id"]
			speaker = markable["speaker"]
			if markable_id not in referent_annotation[chat_id]:
				continue
			if not "unidentifiable" in referent_annotation[chat_id][markable_id] or not referent_annotation[chat_id][markable_id]["unidentifiable"]:
				for referent in referent_annotation[chat_id][markable_id]["referents"]:
					referent_id = referent.split("_")[2]
					referent_color = ent_id2color[referent_id]
					#text2color[markable["text"].lower()].append(referent_color)
					if "black" in word_tokenize(markable["text"]):
						text2color["black"].append(referent_color)
					if "light" in word_tokenize(markable["text"]):
						text2color["light"].append(referent_color)
					if "dark" in word_tokenize(markable["text"]):
						text2color["dark"].append(referent_color)
					if "darker" in word_tokenize(markable["text"]):
						text2color["darker"].append(referent_color)
					if "darkest" in word_tokenize(markable["text"]):
						text2color["darkest"].append(referent_color)
					if "lighter" in word_tokenize(markable["text"]):
						text2color["lighter"].append(referent_color)
					if "lightest" in word_tokenize(markable["text"]):
						text2color["lightest"].append(referent_color)

	plot_colors = sns.color_palette("hls", 8)
	for i, color in enumerate(["black", "darkest", "darker", "dark", "light", "lighter", "lightest"]):
		print("{}: mean {}, std {} (total {})".format(color, np.mean(text2color[color]), np.std(text2color[color]), len(text2color[color])))
		sns.distplot(text2color[color], hist=False, color=plot_colors[i], label=color)
	plt.xlabel('color', fontsize=16)
	plt.ylabel('probability density', fontsize=16)
	plt.legend()
	plt.tight_layout()
	plt.savefig('color_distplot.png', dpi=400)
	plt.clf()

def count_unique_referents(args, dialogue_corpus):
	markable_annotation = read_json("markable_annotation.json")
	referent_annotation = read_json("aggregated_referent_annotation.json")
	aggregated_referent_annotation = read_json("aggregated_referent_annotation.json")

	chat_ids = list(referent_annotation.keys())

	one_player_counter = Counter()
	both_players_counter = Counter()

	for dialogue in dialogue_corpus:
		chat_id = dialogue["uuid"]
		if not chat_id in markable_annotation:
			continue
		else:
			markables = markable_annotation[chat_id]
		if not chat_id in referent_annotation:
			continue

		agent_0_referred = set()
		agent_1_referred = set()

		for markable in markables["markables"]:
			markable_id = markable["markable_id"]
			speaker = markable["speaker"]
			if markable_id not in referent_annotation[chat_id]:
				continue
			if not "unidentifiable" in referent_annotation[chat_id][markable_id] or not referent_annotation[chat_id][markable_id]["unidentifiable"]:
				for referent in referent_annotation[chat_id][markable_id]["referents"]:
					referent_id = referent.split("_")[2]
					if speaker == 0:
						agent_0_referred.add(referent_id)
					else:
						agent_1_referred.add(referent_id)

		one_player_counter[len(agent_0_referred)] += 1
		one_player_counter[len(agent_1_referred)] += 1
		both_players_counter[len(agent_0_referred | agent_1_referred)] += 1
	
	print("one player counter:")
	print(one_player_counter)
	print("\nboth players counter:")
	print(both_players_counter)

	sns.distplot()


def referent_disagreement(args, dialogue_corpus):
	"""
	Compute the agreement of referent resolution based on exact-match/entity-level
	"""
	markable_annotation = read_json("markable_annotation.json")
	referent_annotation = read_json("referent_annotation.json")
	aggregated_referent_annotation = read_json("aggregated_referent_annotation.json")
	chat_ids = list(referent_annotation.keys())

	num_referents_counter = Counter()
	text_counter = Counter()

	num_referents_agreed = Counter()
	text_agreed = Counter()

	num_referents_exact_match = Counter()
	text_exact_match = Counter()

	min_color = 53
	max_color = 203
	min_size = 7
	max_size = 13
	color_bin = 5
	color_range = 1 + int((max_color - min_color) / color_bin)
	size_range = max_size - min_size + 1

	agreed_attributes = np.zeros((color_range, size_range))
	appeared_attributes = np.zeros((color_range, size_range))

	def _group_color(color):
		return int((color - min_color) / color_bin)

	def _group_size(size):
		return size - min_size

	for dialogue in dialogue_corpus:
		chat_id = dialogue["uuid"]
		if not chat_id in markable_annotation:
			continue
		else:
			markables = markable_annotation[chat_id]
		if not chat_id in referent_annotation:
			continue

		mturk_workers = [x for x in list(referent_annotation[chat_id].keys()) if x.startswith("MT_")]
		if len(mturk_workers) < 3:
			continue

		agent_0_kb, agent_1_kb = dialogue["scenario"]["kbs"]
		agent_0_ents = ["agent_0_{}".format(x['id']) for x in agent_0_kb]
		agent_1_ents = ["agent_1_{}".format(x['id']) for x in agent_1_kb]
		ent_id2color = {x['id'] : int(x['color'].split(',')[1]) for x in agent_0_kb + agent_1_kb}
		ent_id2size = {x['id'] : int(x['size']) for x in agent_0_kb + agent_1_kb}

		num_judgement = 0
		num_exact_match = 0
		for markable in markables["markables"]:
			markable_id = markable["markable_id"]
			speaker = markable["speaker"]
			candidate_ents = agent_0_ents if speaker == 0 else agent_1_ents
			if is_annotatable_markable(markable):
				entity_level_agreement = 0
				for annotator in mturk_workers:
					num_referents = len(referent_annotation[chat_id][annotator][markable_id]['referents'])
					num_referents_counter[num_referents] += len(mturk_workers) - 1
					text = markable["text"].lower()
					text_counter[text] += len(mturk_workers) - 1

					for ent in candidate_ents:
						for other_annotator in mturk_workers:
							if annotator == other_annotator:
								continue
							if (ent in referent_annotation[chat_id][annotator][markable_id]["referents"]) == (ent in referent_annotation[chat_id][other_annotator][markable_id]["referents"]):
								num_referents_agreed[num_referents] += 1
								text_agreed[text] += 1

					for other_annotator in mturk_workers:
						if annotator == other_annotator:
							continue
						if set(referent_annotation[chat_id][annotator][markable_id]["referents"]) == set(referent_annotation[chat_id][other_annotator][markable_id]["referents"]):
							num_referents_exact_match[num_referents] += 1
							text_exact_match[text] += 1

				for ent in candidate_ents:
					referent_id = ent.split("_")[2]
					color = _group_color(ent_id2color[referent_id])
					size = _group_size(ent_id2size[referent_id])
					is_referent = 0
					is_not_referent = 0
					for annotator in mturk_workers:
						if ent in referent_annotation[chat_id][annotator][markable_id]["referents"]:
							is_referent += 1
						else:
							is_not_referent += 1
					pairwise_agreement = is_referent * (is_referent - 1) / 2
					pairwise_agreement += is_not_referent * (is_not_referent - 1) / 2
					pairwise_judgements = len(mturk_workers) * (len(mturk_workers) - 1) / 2
					agreed_attributes[color][size] += pairwise_agreement
					appeared_attributes[color][size] += pairwise_judgements


	for k in num_referents_counter.keys():
		print("{}: exact match {:.5f} | entity level agreement {:.5f}".format(k, num_referents_exact_match[k] / num_referents_counter[k], num_referents_agreed[k] / (7 * num_referents_counter[k])))

	text_agreement = {}
	text_exact_match_rate = {}
	for k, _ in text_counter.items(): #.most_common(500):
		text_agreement[k] = text_agreed[k] / (7 * text_counter[k])
		text_exact_match_rate[k] = text_exact_match[k] / text_counter[k]


	unigram_counter = Counter()
	bigram_counter = Counter()
	for text in text_counter.keys():
		for tok in word_tokenize(text.lower()):
			unigram_counter[tok] += 1
		for bigram in bigrams(word_tokenize(text.lower())):
			bigram_counter[bigram] += 1
	tokens = list(unigram_counter.keys())
	#bigram_list = list(bigram_set)

	token_existence = defaultdict(list)
	entity_level_agreement = []
	exact_match_agreement = []

	for text, count in text_counter.items():
		appeared_tokens = word_tokenize(text.lower())
		for tok in tokens:
			if tok in appeared_tokens:
				token_existence[tok] += [1] * count
			else:
				token_existence[tok] += [0] * count
		entity_level_agreement += [text_agreement[text]] * count
		exact_match_agreement += [text_exact_match_rate[text]] * count

	token_agreement_correlation = {}
	for tok in tokens:
		token_agreement_correlation[tok] = np.corrcoef(token_existence[tok], exact_match_agreement)[0][1]

	with open("correlation_list.txt", mode="w") as f:
		f.write("Token & Correlation & Count\n")
		i = 0
		for tok, correlation in sorted(token_agreement_correlation.items(), key=lambda x: x[1]):
			if sum(token_existence[tok]) > 100:
				i += 1
				print("{} & {} & {:.5f} & {} \\\\".format(i, tok, correlation, sum(token_existence[tok])))
				f.write("\\textit{{{}}} & {:.5f} & {} \\\\\n".format(tok, correlation, sum(token_existence[tok])))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--scenario_file', type=str, default="aaai_train_scenarios.json")
	parser.add_argument('--scenario_file_2', type=str, default="aaai_train_scenarios_2.json")
	parser.add_argument('--transcript_file', type=str, default="final_transcripts.json")
	parser.add_argument('--correct_misspellings', action='store_true', default=False)
	parser.add_argument('--replace_strings', action='store_true', default=False)
	parser.add_argument('--success_only', action='store_true', default=False)

	parser.add_argument('--output_brat_format', action='store_true', default=False)
	parser.add_argument('--output_markable_annotation', action='store_true', default=False)
	parser.add_argument('--markable_agreement', action='store_true', default=False)
	parser.add_argument('--fix_agreement', action='store_true', default=False)
	parser.add_argument('--batch_id', type=str, default="batch_00")
	parser.add_argument('--referent_agreement', action='store_true', default=False)
	parser.add_argument('--referent_aggregation', action='store_true', default=False)
	parser.add_argument('--referent_color', action='store_true', default=False)
	parser.add_argument('--count_unique_referents', action='store_true', default=False)
	parser.add_argument('--referent_disagreement', action='store_true', default=False)
	args = parser.parse_args()

	dialogue_corpus = read_json(args.transcript_file)
	scenario_list = read_json(args.scenario_file)
	scenario_list += read_json(args.scenario_file_2)

	if args.output_brat_format:
		dialogue_corpus = output_brat_format(args, dialogue_corpus)

	if args.output_markable_annotation:
		output_markable_annotation(args, dialogue_corpus, fix_misspellings=True)

	if args.markable_agreement:
		markable_agreement(args, dialogue_corpus, annotators=["annotator_1", "annotator_2", "annotator_3"], include_errors=True)

	if args.referent_agreement:
		referent_agreement(args, dialogue_corpus)

	if args.referent_aggregation:
		referent_aggregation(args, dialogue_corpus)

	if args.referent_color:
		referent_color(args, dialogue_corpus)

	if args.count_unique_referents:
		count_unique_referents(args, dialogue_corpus)

	if args.referent_disagreement:
		referent_disagreement(args, dialogue_corpus)
