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
import shutil

import difflib

from nltk import word_tokenize, pos_tag, bigrams, ngrams

from canonical_relations import canonical_relations as canonical_relations_dict
from canonical_functions import canonical_functions as canonical_functions_dict

import pdb
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
sns.set(font_scale=1.15)

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
		self.label = label
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

def span_agreement(args, dialogue_corpus, annotators):
	"""
		Compute span agreement based on Cohen's Kappa.
		Agreement is calculated at the token level judgements.
	"""

	def _add_span(annotation, start_idxs, span_start_idx, span_end_idx, add_zero=False):
		assert len(annotation) == len(start_idxs)
		assert span_start_idx <= span_end_idx

		for i in range(len(start_idxs)):
			if i == len(start_idxs) - 1:
				if start_idxs[i] <= span_start_idx:
					if add_zero:
						annotation[i] = 0
					else:
						annotation[i] = 1
					break

			if start_idxs[i] <= span_start_idx and span_start_idx < start_idxs[i+1]:
				for j in range(i, len(start_idxs)):
					if span_end_idx < start_idxs[j]:
						return annotation				
					if add_zero:
						annotation[j] = 0
					else:
						annotation[j] = 1
		return annotation

	def _compute_agreement(chat_ids, annotator_1, annotator_2, valid_mask):
		total_agreed = 0
		total_valid = 0
		total_annotator_1_positive = 0
		total_annotator_2_positive = 0

		for chat_id in chat_ids:
			assert len(annotator_1[chat_id]) == len(annotator_2[chat_id])
			assert len(annotator_1[chat_id]) == len(valid_mask[chat_id])

			num_agreed = 0
			num_valid = sum(valid_mask[chat_id])
			num_annotator_1_positive = 0
			num_annotator_2_positive = 0
			for i in range(len(annotator_1[chat_id])):
				if valid_mask[chat_id][i]:
					if annotator_1[chat_id][i] == annotator_2[chat_id][i]:
						num_agreed += 1
					if annotator_1[chat_id][i]:
						num_annotator_1_positive += 1
					if annotator_2[chat_id][i]:
						num_annotator_2_positive += 1

			total_agreed += num_agreed
			total_valid += num_valid
			total_annotator_1_positive += num_annotator_1_positive
			total_annotator_2_positive += num_annotator_2_positive

		observed_agreement = total_agreed / total_valid
		
		annotator_1_positive_prob = total_annotator_1_positive / total_valid
		annotator_2_positive_prob = total_annotator_2_positive / total_valid
		expected_agreement = annotator_1_positive_prob * annotator_2_positive_prob + \
						(1 - annotator_1_positive_prob) * (1 - annotator_2_positive_prob)

		cohens_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

		print("total chats: {}".format(len(chat_ids)))
		print("total judgements: {}".format(total_valid))
		print("observed agreement: {}".format(observed_agreement))
		print("expected agreement: {}".format(expected_agreement))
		print("Cohen's Kappa: {}".format(cohens_kappa))

	markable_annotation = read_json("markable_annotation.json")
	relation_span = {}
	attribute_span = {}
	modifier_span = {}
	relation_start = {}
	attribute_start = {}
	modifier_start = {}
	utterance_mask = {}
	outside_markable_mask = {}
	chat_ids = set()

	for filename in glob.glob('span_detection/annotator_1/batch_05/*.ann'):
		chat_id = filename.split("/")[3].split(".")[0]
		chat_ids.add(chat_id)

	for annotator in annotators:
		relation_span[annotator] = {}
		attribute_span[annotator] = {}
		modifier_span[annotator] = {}
		relation_start[annotator] = {}
		attribute_start[annotator] = {}
		modifier_start[annotator] = {}

		for filename in glob.glob('span_detection/' + annotator + '/batch_05/*.ann'):
			chat_id = filename.split("/")[3].split(".")[0]

			tokens = []
			utterance_mask[chat_id] = []
			outside_markable_mask[chat_id] = []
			start_idxs = []

			# compute candidate_tokens, utterance_mask, start_idxs
			text = markable_annotation[chat_id]["text"]			
			start_idx = 0
			for line in text.split("\n"):
				utterance_tokens = line.split(" ")
				tokens += utterance_tokens
				utterance_mask[chat_id].append(0)
				utterance_mask[chat_id] += [1] * (len(utterance_tokens) - 1)
				for tok in line.split(" "):
					start_idxs.append(start_idx)
					start_idx += len(tok) + 1

			# compute outside_markable_mask
			outside_markable_mask[chat_id] = copy.copy(utterance_mask[chat_id])
			for markable in markable_annotation[chat_id]["markables"]:
				if not markable["generic"] and (markable["predicative"] is None):
					span_start_idx = markable["start"]
					span_end_idx = markable["end"]
					outside_markable_mask[chat_id] = _add_span(outside_markable_mask[chat_id], start_idxs, span_start_idx, span_end_idx, add_zero=True)

			# compute brat_ids of split relations
			split_brat_id = set()
			with open(filename, "r") as fin:
				ann = fin.read()
				for line in ann.split("\n"):
					brat_id = line.split("\t")[0]
					if brat_id.startswith("R"):
						label = line.split("\t")[1].split(" ")[0]
						if label == "Split":
							arg1 = line.split("\t")[1].split(" ")[1].split(":")[1]
							arg2 = line.split("\t")[1].split(" ")[2].split(":")[1]
							split_brat_id.add(arg1)

			# compute relation, attribute, modifier spans
			relation_span[annotator][chat_id] = [0] * len(tokens)
			attribute_span[annotator][chat_id] = [0] * len(tokens)
			modifier_span[annotator][chat_id] = [0] * len(tokens)
			relation_start[annotator][chat_id] = [0] * len(tokens)
			attribute_start[annotator][chat_id] = [0] * len(tokens)
			modifier_start[annotator][chat_id] = [0] * len(tokens)

			with open(filename, "r") as fin:
				ann = fin.read()
				for line in ann.split("\n"):
					brat_id = line.split("\t")[0]
					if brat_id.startswith("T"):
						label = line.split("\t")[1].split(" ")[0]
						# compute relation span
						if label in ["Spatial-Relation", "Spatial-Relation-Markable"]:
							span_start_idx = int(line.split("\t")[1].split(" ")[1])
							span_end_idx = int(line.split("\t")[1].split(" ")[2])
							relation_span[annotator][chat_id] = _add_span(relation_span[annotator][chat_id],
																start_idxs, span_start_idx, span_end_idx)
							if not brat_id in split_brat_id:
								relation_start[annotator][chat_id] = _add_span(relation_start[annotator][chat_id],
																	start_idxs, span_start_idx, span_start_idx)

						if label == "Spatial-Attribute":
							span_start_idx = int(line.split("\t")[1].split(" ")[1])
							span_end_idx = int(line.split("\t")[1].split(" ")[2])
							attribute_span[annotator][chat_id] = _add_span(attribute_span[annotator][chat_id],
																start_idxs, span_start_idx, span_end_idx)
							attribute_start[annotator][chat_id] = _add_span(attribute_span[annotator][chat_id],
																start_idxs, span_start_idx, span_start_idx)
						if label == "Modifier":
							span_start_idx = int(line.split("\t")[1].split(" ")[1])
							span_end_idx = int(line.split("\t")[1].split(" ")[2])
							modifier_span[annotator][chat_id] = _add_span(modifier_span[annotator][chat_id],
																start_idxs, span_start_idx, span_end_idx)
							modifier_start[annotator][chat_id] = _add_span(attribute_span[annotator][chat_id],
																start_idxs, span_start_idx, span_start_idx)


	# compute relation agreement
	print("relation span agreement")
	_compute_agreement(chat_ids, relation_span["annotator_1"], relation_span["annotator_2"], utterance_mask)
	print("")

	# compute attribute agreement
	print("attribute span agreement")
	_compute_agreement(chat_ids, attribute_span["annotator_1"], attribute_span["annotator_2"], outside_markable_mask)
	print("")

	# compute modifier agreement
	print("modifier span agreement")
	_compute_agreement(chat_ids, modifier_span["annotator_1"], modifier_span["annotator_2"], utterance_mask)
	print("")

	# compute relation start agreement
	print("relation start agreement")
	_compute_agreement(chat_ids, relation_start["annotator_1"], relation_start["annotator_2"], utterance_mask)
	print("")

	# compute attribute start agreement
	print("attribute start agreement")
	_compute_agreement(chat_ids, attribute_start["annotator_1"], attribute_start["annotator_2"], outside_markable_mask)
	print("")

	# compute modifier start agreement
	print("modifier start agreement")
	_compute_agreement(chat_ids, modifier_start["annotator_1"], modifier_start["annotator_2"], utterance_mask)
	print("")

def argument_agreement(args, dialogue_corpus, annotators):
	"""
		1. Compute exact match agreement (based on Cohen's Kappa)
		2. Compute essential agreement rate (referents match)
	"""
	def _compute_agreement(total_pairwise_judgements, total_pairwise_disagreement, total_annotator_1_arguments, total_annotator_2_arguments):
		observed_agreement = (total_pairwise_judgements - total_pairwise_disagreement) / total_pairwise_judgements
		annotator_1_argument_prob = total_annotator_1_arguments / total_pairwise_judgements
		annotator_2_argument_prob = total_annotator_2_arguments / total_pairwise_judgements
		expected_agreement = annotator_1_argument_prob * annotator_2_argument_prob + (1 - annotator_1_argument_prob) * (1 - annotator_2_argument_prob)
		cohens_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
		
		print("total judgements: {}".format(total_pairwise_judgements))
		print("observed agreement: {}".format(observed_agreement))
		print("expected agreement: {}".format(expected_agreement))
		print("Cohen's Kappa: {}".format(cohens_kappa))	

	markable_annotation = read_json("markable_annotation.json")
	referent_annotation = read_json("aggregated_referent_annotation.json")
	annotator_1 = read_json("annotator_1.json")
	annotator_2 = read_json("annotator_2.json")

	chat_ids = list(annotator_1.keys())
	assert chat_ids == list(annotator_2.keys())

	# compute utterance_end_idxs
	utterance_end_idxs = {}
	for chat_id in chat_ids:
		utterance_end_idxs[chat_id] = []
		text = markable_annotation[chat_id]["text"]
		current_len = 0
		for line in text.split("\n"):
			current_len += len(line) + 1
			utterance_end_idxs[chat_id].append(current_len)
		assert current_len - 1 == len(text)

	# Step 1: Compute subject argument agreement
	total_pairwise_judgements = 0
	total_pairwise_disagreement = 0
	total_annotator_1_arguments = 0
	total_annotator_2_arguments = 0

	essential_agreement = 0
	essential_disagreement = 0

	for chat_id in chat_ids:
		# compute number of candidate markables in first k utterances
		num_candidates = []
		for k in range(len(utterance_end_idxs[chat_id])):
			num_canditates_k = 0
			for markable in markable_annotation[chat_id]["markables"]:
				if not markable["generic"] and (markable["predicative"] is None) and markable["start"] < utterance_end_idxs[chat_id][k]:
					num_canditates_k += 1
			num_candidates.append(num_canditates_k)

		# compute num_pairwise_judgements, num_pairwise_agreement, num_annotator_arguments
		num_pairwise_judgements = 0
		num_pairwise_disagreement = 0
		num_annotator_1_arguments = 0
		num_annotator_2_arguments = 0

		for i in range(len(annotator_1[chat_id]["relations"])):
			annotator_1_relation = annotator_1[chat_id]["relations"][i]
			annotator_2_relation = annotator_2[chat_id]["relations"][i]

			if "is_split" in annotator_1_relation["tags"]:
				continue

			for k in range(len(utterance_end_idxs[chat_id])):
				if annotator_1_relation["start"] < utterance_end_idxs[chat_id][k]:
					num_pairwise_judgements += num_candidates[k]
					break

			is_disagreement = False
			for subject in annotator_1_relation["subjects"]:
				if subject not in annotator_2_relation["subjects"]:
					num_pairwise_disagreement += 1
					is_disagreement = True
				num_annotator_1_arguments += 1
			for subject in annotator_2_relation["subjects"]:
				if subject not in annotator_1_relation["subjects"]:
					num_pairwise_disagreement += 1
					is_disagreement = True
				num_annotator_2_arguments += 1

			if is_disagreement:
				annotator_1_referents = set()
				for subject in annotator_1_relation["subjects"]:
					for referent in referent_annotation[chat_id][subject]["referents"]:
						referent_id = int(referent.split('_')[-1])
						annotator_1_referents.add(referent_id)
				annotator_2_referents = set()
				for subject in annotator_2_relation["subjects"]:
					for referent in referent_annotation[chat_id][subject]["referents"]:
						referent_id = int(referent.split('_')[-1])
						annotator_2_referents.add(referent_id)
				if annotator_1_referents == annotator_2_referents:
					essential_agreement += 1
				else:
					essential_disagreement += 1

		assert len(annotator_1[chat_id]["attributes"]) == len(annotator_2[chat_id]["attributes"])
		for i in range(len(annotator_1[chat_id]["attributes"])):
			annotator_1_attribute = annotator_1[chat_id]["attributes"][i]
			annotator_2_attribute = annotator_2[chat_id]["attributes"][i]

			for k in range(len(utterance_end_idxs[chat_id])):
				if annotator_1_attribute["start"] < utterance_end_idxs[chat_id][k]:
					num_pairwise_judgements += num_candidates[k]
					break

			for subject in annotator_1_attribute["subjects"]:
				if subject not in annotator_2_attribute["subjects"]:
					num_pairwise_disagreement += 1
				num_annotator_1_arguments += 1
			for subject in annotator_2_attribute["subjects"]:
				if subject not in annotator_1_attribute["subjects"]:
					num_pairwise_disagreement += 1
				num_annotator_2_arguments += 1

		total_pairwise_judgements += num_pairwise_judgements
		total_pairwise_disagreement += num_pairwise_disagreement
		total_annotator_1_arguments += num_annotator_1_arguments
		total_annotator_2_arguments += num_annotator_2_arguments

	# print subject agreement
	print("===subject agreement===")
	_compute_agreement(total_pairwise_judgements, total_pairwise_disagreement, total_annotator_1_arguments, total_annotator_2_arguments)
	if (essential_agreement + essential_disagreement) > 0:
		print("essential agreement: {:.2f}% ({} out of {})".format(100.0 * essential_agreement / (essential_agreement + essential_disagreement),
																	essential_agreement, (essential_agreement + essential_disagreement)))

	# Step 2: Compute object argument agreement
	total_pairwise_judgements = 0
	total_pairwise_disagreement = 0
	total_annotator_1_arguments = 0
	total_annotator_2_arguments = 0

	essential_agreement = 0
	essential_disagreement = 0

	for chat_id in chat_ids:
		# compute number of candidate markables in first k utterances
		num_candidates = []
		for k in range(len(utterance_end_idxs[chat_id])):
			num_canditates_k = 0
			for markable in markable_annotation[chat_id]["markables"]:
				if not markable["generic"] and (markable["predicative"] is None) and markable["start"] < utterance_end_idxs[chat_id][k]:
					num_canditates_k += 1
			num_candidates.append(num_canditates_k)

		# compute num_pairwise_judgements, num_pairwise_agreement, num_annotator_arguments
		num_pairwise_judgements = 0
		num_pairwise_disagreement = 0
		num_annotator_1_arguments = 0
		num_annotator_2_arguments = 0

		assert len(annotator_1[chat_id]["relations"]) == len(annotator_2[chat_id]["relations"])
		for i in range(len(annotator_1[chat_id]["relations"])):
			annotator_1_relation = annotator_1[chat_id]["relations"][i]
			annotator_2_relation = annotator_2[chat_id]["relations"][i]

			if "is_split" in annotator_1_relation["tags"]:
				continue

			for k in range(len(utterance_end_idxs[chat_id])):
				if annotator_1_relation["start"] < utterance_end_idxs[chat_id][k]:
					num_pairwise_judgements += num_candidates[k]
					break

			is_disagreement = False
			for obj in annotator_1_relation["objects"]:
				if obj not in annotator_2_relation["objects"]:
					num_pairwise_disagreement += 1
					is_disagreement = True
				num_annotator_1_arguments += 1
			for obj in annotator_2_relation["objects"]:
				if obj not in annotator_1_relation["objects"]:
					num_pairwise_disagreement += 1
					is_disagreement = True
				num_annotator_2_arguments += 1

			if is_disagreement:
				annotator_1_referents = set()
				for obj in annotator_1_relation["objects"]:
					for referent in referent_annotation[chat_id][obj]["referents"]:
						referent_id = int(referent.split('_')[-1])
						annotator_1_referents.add(referent_id)
				for subject in annotator_1_relation["subjects"]:
					for referent in referent_annotation[chat_id][subject]["referents"]:
						referent_id = int(referent.split('_')[-1])
						if referent_id in annotator_1_referents:
							annotator_1_referents.remove(referent_id)

				annotator_2_referents = set()
				for obj in annotator_2_relation["objects"]:
					for referent in referent_annotation[chat_id][obj]["referents"]:
						referent_id = int(referent.split('_')[-1])
						annotator_2_referents.add(referent_id)
				for subject in annotator_2_relation["subjects"]:
					for referent in referent_annotation[chat_id][subject]["referents"]:
						referent_id = int(referent.split('_')[-1])
						if referent_id in annotator_2_referents:
							annotator_2_referents.remove(referent_id)

				if annotator_1_referents == annotator_2_referents:
					essential_agreement += 1
				else:
					essential_disagreement += 1

		total_pairwise_judgements += num_pairwise_judgements
		total_pairwise_disagreement += num_pairwise_disagreement
		total_annotator_1_arguments += num_annotator_1_arguments
		total_annotator_2_arguments += num_annotator_2_arguments

	# compute object agreement
	print("===object agreement===")
	_compute_agreement(total_pairwise_judgements, total_pairwise_disagreement, total_annotator_1_arguments, total_annotator_2_arguments)
	if (essential_agreement + essential_disagreement) > 0:
		print("essential agreement: {:.2f}% ({} out of {})".format(100.0 * essential_agreement / (essential_agreement + essential_disagreement),
																	essential_agreement, (essential_agreement + essential_disagreement)))


	# Step 3: Compute modificant agreement
	total_pairwise_judgements = 0
	total_pairwise_disagreement = 0
	total_annotator_1_modificants = 0
	total_annotator_2_modificants = 0

	for chat_id in chat_ids:
		# compute number of candidate spatial expressions in first k utterances
		num_candidates = []
		for k in range(len(utterance_end_idxs[chat_id])):
			num_canditates_k = 0

			for relation in annotator_1[chat_id]["relations"]:
				if relation["start"] < utterance_end_idxs[chat_id][k]:
					num_canditates_k += 1

			for attribute in annotator_1[chat_id]["attributes"]:
				if attribute["start"] < utterance_end_idxs[chat_id][k]:
					num_canditates_k += 1

			num_candidates.append(num_canditates_k)

		# collect modifier id --> modificants
		annotator_1_modificants = {}
		annotator_2_modificants = {}
		# collect modifier id --> modifier start
		modifier_id2start = {}
		assert len(annotator_1[chat_id]["relations"]) == len(annotator_2[chat_id]["relations"])
		for i in range(len(annotator_1[chat_id]["relations"])):
			annotator_1_relation = annotator_1[chat_id]["relations"][i]
			for modifier in annotator_1_relation["modifiers"]:
				annotator_1_modificants[modifier["id"]] = "relation_{}".format(i)
				modifier_id2start[modifier["id"]] = modifier["start"]
			annotator_2_relation = annotator_2[chat_id]["relations"][i]
			for modifier in annotator_2_relation["modifiers"]:
				annotator_2_modificants[modifier["id"]] = "relation_{}".format(i)
				modifier_id2start[modifier["id"]] = modifier["start"]
		assert len(annotator_1[chat_id]["attributes"]) == len(annotator_2[chat_id]["attributes"])
		for i in range(len(annotator_1[chat_id]["attributes"])):
			annotator_1_attribute = annotator_1[chat_id]["attributes"][i]
			for modifier in annotator_1_attribute["modifiers"]:
				annotator_1_modificants[modifier["id"]] = "attribute_{}".format(i)
				modifier_id2start[modifier["id"]] = modifier["start"]
			annotator_2_attribute = annotator_2[chat_id]["attributes"][i]
			for modifier in annotator_2_attribute["modifiers"]:
				annotator_2_modificants[modifier["id"]] = "attribute_{}".format(i)
				modifier_id2start[modifier["id"]] = modifier["start"]

		# compute num_pairwise_judgements, num_pairwise_agreement, num_annotator_arguments
		num_pairwise_judgements = 0
		num_pairwise_disagreement = 0
		num_annotator_1_modificants = len(annotator_1_modificants)
		num_annotator_2_modificants = len(annotator_2_modificants)
		assert num_annotator_1_modificants == num_annotator_2_modificants

		for modifier_id in annotator_1_modificants.keys():
			for k in range(len(utterance_end_idxs[chat_id])):
				if modifier_id2start[modifier_id] < utterance_end_idxs[chat_id][k]:
					num_pairwise_judgements += num_candidates[k]
					break

			if annotator_1_modificants[modifier_id] != annotator_2_modificants[modifier_id]:
				num_pairwise_disagreement += 2

		total_pairwise_judgements += num_pairwise_judgements
		total_pairwise_disagreement += num_pairwise_disagreement
		total_annotator_1_modificants += num_annotator_1_modificants
		total_annotator_2_modificants += num_annotator_2_modificants

	# compute modifier agreement
	print("===modifier agreement===")
	_compute_agreement(total_pairwise_judgements, total_pairwise_disagreement, total_annotator_1_modificants, total_annotator_2_modificants)


def canonical_agreement(args, dialogue_corpus, annotators):
	"""
		1. Compute Cohen's Kappa for canonical relation
		2. Compute Cohen's Kappa for canonical function
	"""
	markable_annotation = read_json("markable_annotation.json")
	referent_annotation = read_json("aggregated_referent_annotation.json")
	annotator_1 = read_json("annotator_1.json")
	annotator_2 = read_json("annotator_2.json")

	chat_ids = list(annotator_1.keys())
	assert chat_ids == list(annotator_2.keys())

	# Step 1. Compute canonical relation agreement
	canonical_relations = []
	for canonical_relation_category in canonical_relations_dict.keys():
		canonical_relations += list(canonical_relations_dict[canonical_relation_category].keys())

	annotator_1_canonical_count = Counter()
	annotator_2_canonical_count = Counter()

	total_relations = 0
	total_disagreement = 0
	for chat_id in chat_ids:
		assert len(annotator_1[chat_id]["relations"]) == len(annotator_2[chat_id]["relations"])
		for i in range(len(annotator_1[chat_id]["relations"])):
			annotator_1_relation = annotator_1[chat_id]["relations"][i]
			annotator_2_relation = annotator_2[chat_id]["relations"][i]

			for canonical_relation in annotator_1_relation["canonical-relations"]:
				if canonical_relation == "undefined":
					continue
				annotator_1_canonical_count[canonical_relation] += 1
				if canonical_relation not in annotator_2_relation["canonical-relations"]:
					total_disagreement += 1
			for canonical_relation in annotator_2_relation["canonical-relations"]:
				if canonical_relation == "undefined":
					continue
				annotator_2_canonical_count[canonical_relation] += 1
				if canonical_relation not in annotator_1_relation["canonical-relations"]:
					total_disagreement += 1 

			total_relations += 1

	total_judgements = 	total_relations * len(canonical_relations)
	observed_agreement = (total_judgements - total_disagreement) / total_judgements
	expected_agreements = []
	for canonical_relation in canonical_relations:
		annotator_1_prob = annotator_1_canonical_count[canonical_relation] / total_relations
		annotator_2_prob = annotator_2_canonical_count[canonical_relation] / total_relations
		expected_agreements.append(annotator_1_prob * annotator_2_prob + (1 - annotator_1_prob) * (1 - annotator_2_prob))
	expected_agreement = np.mean(expected_agreements)
	cohens_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)		
	print("total judgements: {}".format(total_judgements))
	print("observed agreement: {}".format(observed_agreement))
	print("expected agreement: {}".format(expected_agreement))
	print("Cohen's Kappa: {}".format(cohens_kappa))	

	# Step 2. Compute canonical function agreement

	canonical_functions = []
	for canonical_function in canonical_functions_dict.keys():
		canonical_functions.append(canonical_function)

	annotator_1_canonical_count = Counter()
	annotator_2_canonical_count = Counter()

	total_modifiers = 0
	total_agreement = 0
	for chat_id in chat_ids:
		assert len(annotator_1[chat_id]["relations"]) == len(annotator_2[chat_id]["relations"])
		for i in range(len(annotator_1[chat_id]["relations"])):
			annotator_1_relation = annotator_1[chat_id]["relations"][i]
			annotator_1_id2canonical = {}
			for modifier in annotator_1_relation["modifiers"]:
				annotator_1_id2canonical[modifier["id"]] = modifier["canonical-function"]

			annotator_2_relation = annotator_2[chat_id]["relations"][i]
			annotator_2_id2canonical = {}
			for modifier in annotator_2_relation["modifiers"]:
				annotator_2_id2canonical[modifier["id"]] = modifier["canonical-function"]

			assert annotator_1_id2canonical.keys() == annotator_2_id2canonical.keys()
			for modifier_id in annotator_1_id2canonical.keys():
				if annotator_1_id2canonical[modifier_id] == annotator_2_id2canonical[modifier_id]:
					total_agreement += 1
				annotator_1_canonical_count[annotator_1_id2canonical[modifier_id]] += 1
				annotator_2_canonical_count[annotator_2_id2canonical[modifier_id]] += 1
				total_modifiers += 1

	total_judgements = 	total_modifiers
	observed_agreement = total_agreement / total_judgements
	expected_agreement = 0
	for canonical_function in canonical_functions:
		expected_agreement += (annotator_1_canonical_count[canonical_function] / total_modifiers) * (annotator_2_canonical_count[canonical_function] / total_modifiers)
	cohens_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)		
	print("total judgements: {}".format(total_judgements))
	print("observed agreement: {}".format(observed_agreement))
	print("expected agreement: {}".format(expected_agreement))
	print("Cohen's Kappa: {}".format(cohens_kappa))	

	pdb.set_trace()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=1, help='random seed')
	parser.add_argument('--scenario_file', type=str, default="aaai_train_scenarios.json")
	parser.add_argument('--scenario_file_2', type=str, default="aaai_train_scenarios_2.json")
	parser.add_argument('--transcript_file', type=str, default="final_transcripts.json")

	parser.add_argument('--span_agreement', action='store_true', default=False)
	parser.add_argument('--argument_agreement', action='store_true', default=False)
	parser.add_argument('--canonical_agreement', action='store_true', default=False)

	args = parser.parse_args()

	np.random.seed(args.seed)

	dialogue_corpus = read_json(args.transcript_file)
	scenario_list = read_json(args.scenario_file)
	scenario_list += read_json(args.scenario_file_2)

	if args.span_agreement:
		span_agreement(args, dialogue_corpus, annotators=["annotator_1", "annotator_2"])

	if args.argument_agreement:
		argument_agreement(args, dialogue_corpus, annotators=["annotator_1", "annotator_2"])

	if args.canonical_agreement:
		canonical_agreement(args, dialogue_corpus, annotators=["annotator_1", "annotator_2"])

