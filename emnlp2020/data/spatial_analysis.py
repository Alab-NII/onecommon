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

from canonical_relations import canonical_relations, detect_canonical_relations, check_canonical_relation
from canonical_functions import canonical_functions, detect_canonical_function

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

def output_brat_format(args, dialogue_corpus, sample_size=600, batch_size=100):
	markable_annotation = read_json("markable_annotation.json")

	finished_batches = [x for x in os.listdir('annotated') if x.startswith('batch_')]
	batch_info = defaultdict(list)
	chat_id2batch_id = {}
	
	chat_ids = list(markable_annotation.keys())

	# compute batch_ids of each chat_ids
	for finished_batch in finished_batches:
		for filename in glob.glob('annotated/' + finished_batch + '/*.txt'):
			chat_id = filename.split("/")[2].split(".")[0]
			batch_info[finished_batch].append(chat_id)
			chat_id2batch_id[chat_id] = finished_batch

	sampled_chat_ids = np.random.choice(chat_ids, size=sample_size, replace=False)
	current_batch_id = 0
	current_batch_size = 0
	for sampled_chat_id in sampled_chat_ids:
		current_batch_size += 1
		if not os.path.exists("spatial_annotation/batch_{:0>2}".format(current_batch_id)):
			os.makedirs("spatial_annotation/batch_{:0>2}".format(current_batch_id))
		if not os.path.exists("spatial_annotation/modified/batch_{:0>2}".format(current_batch_id)):
			os.makedirs("spatial_annotation/modified/batch_{:0>2}".format(current_batch_id))

		shutil.copy("annotated/" + chat_id2batch_id[sampled_chat_id] + "/{}.txt".format(sampled_chat_id),
					"spatial_annotation/batch_{:0>2}/{}.txt".format(current_batch_id, sampled_chat_id))
		shutil.copy("annotated/" + chat_id2batch_id[sampled_chat_id] + "/{}.ann".format(sampled_chat_id),
					"spatial_annotation/batch_{:0>2}/{}.ann".format(current_batch_id, sampled_chat_id))

		with open("annotated/" + chat_id2batch_id[sampled_chat_id] + "/{}.txt".format(sampled_chat_id), "r") as fin:
			original_text = fin.read()

		if original_text != markable_annotation[sampled_chat_id]["text"]:
			with open("spatial_annotation/modified/batch_{:0>2}/{}.txt".format(current_batch_id, sampled_chat_id), "w") as fout:
				fout.write(markable_annotation[sampled_chat_id]["text"])

			with open("spatial_annotation/modified/batch_{:0>2}/{}.ann".format(current_batch_id, sampled_chat_id), "w") as fout:
				fout.write("")

		if current_batch_size % batch_size == 0:
			current_batch_id += 1
			current_batch_size = 0

def output_spatial_annotation(args, dialogue_corpus):
	markable_annotation = read_json("markable_annotation.json")
	spatial_annotation = {}

	if args.annotator:
		filepath = args.annotator
	else:
		filepath = "annotated"

	annotated_batches = [x for x in os.listdir('spatial_annotation/' + filepath) if x.startswith('batch_')]
	batch_info = defaultdict(list)
	chat_id2batch_id = {}

	for batch_id in annotated_batches:
		for filename in glob.glob('spatial_annotation/' + filepath + '/' + batch_id + '/*.ann'):
			chat_id = filename.split("/")[3].split(".")[0]
			chat_id2batch_id[chat_id] = batch_id
			spatial_annotation[chat_id] = {}
			spatial_annotation[chat_id]["relations"] = []
			spatial_annotation[chat_id]["attributes"] = []

			with open('spatial_annotation/' + filepath + '/' + batch_id + '/{}.txt'.format(chat_id), "r") as fin:
				original_text = fin.read()

			if original_text != markable_annotation[chat_id]["text"]:
				print(chat_id)
				assert False

			# read through annotation to collect data
			brat_id2 = {}
			brat_id2markable_id = {}
			brat_id2modifiers = {}
			brat_id2relation_start = {}
			brat_id2attribute_start = {}
			brat_id2relation_end = {}
			brat_id2attribute_end = {}
			explicit_subjects = defaultdict(list)
			explicit_objects = defaultdict(list)
			explicit_modifiers = defaultdict(list)
			explicit_splits = defaultdict(list)

			modifier_id = 0

			# Attributes of spatial relations/attributes/modifiers
			# todo: fix naming (potential confusion of brat attributes and spatial attributes)
			explicit_attributes = defaultdict(list)
			brat_id2annotator_note = {}
			with open(filename, "r") as fin:
				ann = fin.read()
				for line in ann.split("\n"):
					brat_id = line.split("\t")[0]
					if brat_id.startswith("T"):
						label = line.split("\t")[1].split(" ")[0]
						if label in ["Markable", "Spatial-Relation-Markable"]:
							start = int(line.split("\t")[1].split(" ")[1])
							end = int(line.split("\t")[1].split(" ")[2])
							text = line.split("\t")[-1]
							markable_id = None
							for markable in markable_annotation[chat_id]["markables"]:
								if markable["start"] == start or markable["end"] == end:
									markable_id = markable["markable_id"]
									break
							else:
								pdb.set_trace()
							if markable_id:
								brat_id2markable_id[brat_id] = markable_id
						if label in ["Spatial-Relation", "Spatial-Relation-Markable"]:
							start = int(line.split("\t")[1].split(" ")[1])
							end = int(line.split("\t")[1].split(" ")[2])
							brat_id2relation_start[brat_id] = start
							brat_id2relation_end[brat_id] = end
						if label == "Spatial-Attribute":
							start = int(line.split("\t")[1].split(" ")[1])
							end = int(line.split("\t")[1].split(" ")[2])
							brat_id2attribute_start[brat_id] = start
							brat_id2attribute_end[brat_id] = end
						if label == "Modifier":
							modifier = {}
							modifier["start"] = int(line.split("\t")[1].split(" ")[1])
							modifier["end"] = int(line.split("\t")[1].split(" ")[2])
							modifier["text"] = line.split("\t")[-1]
							modifier["id"] = modifier_id
							modifier_id += 1
							brat_id2modifiers[brat_id] = copy.copy(modifier)
					elif brat_id.startswith("R"):
						label = line.split("\t")[1].split(" ")[0]
						arg1 = line.split("\t")[1].split(" ")[1].split(":")[1]
						arg2 = line.split("\t")[1].split(" ")[2].split(":")[1]
						if label == "Subj":
							explicit_subjects[arg1].append(arg2)
						elif label == "Obj":
							explicit_objects[arg1].append(arg2)
						elif label == "Mod":
							explicit_modifiers[arg2].append(arg1)
						elif label == "Split":
							explicit_splits[arg1] = arg2
					elif brat_id.startswith("A"):
						label = line.split("\t")[1].split(" ")[0]
						arg = line.split("\t")[1].split(" ")[1]
						explicit_attributes[arg].append(label)
					elif brat_id.startswith("#"):
						arg = line.split("\t")[1].split(" ")[1]
						note = line.split("\t")[2]
						brat_id2annotator_note[arg] = note

			# concat dictionaries
			brat_id2end = copy.copy(brat_id2relation_end)
			brat_id2end.update(brat_id2attribute_end)

			# re-index relation and attribute ids by start position
			brat_id2relation_id = {}
			brat_id2attribute_id = {}
			for i, item in enumerate(sorted(brat_id2relation_start.items(), key=lambda item: item[1])):
				brat_id = item[0]
				brat_id2relation_id[brat_id] = i
				spatial_annotation[chat_id]["relations"].append({})
			for i, item in enumerate(sorted(brat_id2attribute_start.items(), key=lambda item: item[1])):
				brat_id = item[0]
				brat_id2attribute_id[brat_id] = i
				spatial_annotation[chat_id]["attributes"].append({})

			# Step 1. Collect primary information of spatial annotation
			with open(filename, "r") as fin:
				ann = fin.read()
				for line in ann.split("\n"):
					brat_id = line.split("\t")[0]
					if brat_id.startswith("T"):
						label = line.split("\t")[1].split(" ")[0]
						if label in ["Spatial-Relation", "Spatial-Relation-Markable"]:
							start = int(line.split("\t")[1].split(" ")[1])
							end = int(line.split("\t")[1].split(" ")[2])
							text = line.split("\t")[-1]
							relation_id = brat_id2relation_id[brat_id]

							spatial_annotation[chat_id]["relations"][relation_id]["start"] = start
							spatial_annotation[chat_id]["relations"][relation_id]["end"] = end
							spatial_annotation[chat_id]["relations"][relation_id]["text"] = text
							spatial_annotation[chat_id]["relations"][relation_id]["subjects"] = []
							spatial_annotation[chat_id]["relations"][relation_id]["objects"] = []
							spatial_annotation[chat_id]["relations"][relation_id]["modifiers"] = []
							spatial_annotation[chat_id]["relations"][relation_id]["tags"] = set()
							spatial_annotation[chat_id]["relations"][relation_id]["paraphrase"] = ""
							spatial_annotation[chat_id]["relations"][relation_id]["canonical-relations"] = []
							spatial_annotation[chat_id]["relations"][relation_id]["splits"] = []

							assert markable_annotation[chat_id]["text"][start:end] == text

							# check basic attributes
							if brat_id in explicit_attributes:
								for explicit_attribute in explicit_attributes[brat_id]:
									if explicit_attribute == "No-Obj":
										spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("no_object")
									if explicit_attribute in ["Ambiguous-Subj", "Ambiguous-Obj"]:
										spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("ambiguous")
									if explicit_attribute == "Canonical-Undefined":
										spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("canonical_undefined")
									if explicit_attribute in ["Unannotatable", "No-Subj-Markable", "No-Obj-Markable"]:
										spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("unannotatable")
							if brat_id in explicit_splits:
								spatial_annotation[chat_id]["relations"][brat_id2relation_id[explicit_splits[brat_id]]]["splits"].append(relation_id)
								spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("is_split")

							# add subjects
							if brat_id in explicit_subjects:
								for explicit_subject in explicit_subjects[brat_id]:
									# check if the argument markable exists
									if explicit_subject in brat_id2markable_id:
										spatial_annotation[chat_id]["relations"][relation_id]["subjects"].append(brat_id2markable_id[explicit_subject])
									else:
										spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("unannotatable")
							else:
								if "unannotatable" not in spatial_annotation[chat_id]["relations"][relation_id]["tags"]:
									subject_markable_id = None
									for markable in markable_annotation[chat_id]["markables"]:
										if markable["generic"] or markable["predicative"]:
											continue
										if markable["start"] <= start:
											subject_markable_id = markable["markable_id"]
										else:
											break
									if subject_markable_id:
										spatial_annotation[chat_id]["relations"][relation_id]["subjects"].append(subject_markable_id)

							# add objects
							if brat_id in explicit_objects:
								for explicit_object in explicit_objects[brat_id]:
									if explicit_object in brat_id2markable_id:
										spatial_annotation[chat_id]["relations"][relation_id]["objects"].append(brat_id2markable_id[explicit_object])
									else:
										spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("unannotatable")
							else:
								if "unannotatable" not in spatial_annotation[chat_id]["relations"][relation_id]["tags"] and \
									"no_object" not in spatial_annotation[chat_id]["relations"][relation_id]["tags"]:
									object_markable_id = None
									for mi, markable in enumerate(markable_annotation[chat_id]["markables"]):
										if markable["generic"] or markable["predicative"]:
											continue
										if markable["start"] > start:
											object_markable_id =   markable["markable_id"]
											break
									if object_markable_id:
										spatial_annotation[chat_id]["relations"][relation_id]["objects"].append(object_markable_id)

							# add explicit modifiers
							if brat_id in explicit_modifiers:
								for explicit_modifier in explicit_modifiers[brat_id]:
									modifier = brat_id2modifiers[explicit_modifier]
									if explicit_modifier in explicit_attributes:
										canonical_function = explicit_attributes[explicit_modifier][0]
										if canonical_function == "Mod-Subtlty":
											canonical_function = "Mod-Subtlety"
									else:
										canonical_function = detect_canonical_function(modifier["text"])
									modifier["canonical-function"] = canonical_function

									spatial_annotation[chat_id]["relations"][relation_id]["modifiers"].append(modifier)

							# add annotator_note
							if brat_id in brat_id2annotator_note:
								spatial_annotation[chat_id]["relations"][relation_id]["paraphrase"] = brat_id2annotator_note[brat_id]

							# change from set to list
							spatial_annotation[chat_id]["relations"][relation_id]["tags"] = list(spatial_annotation[chat_id]["relations"][relation_id]["tags"])

						elif label == "Spatial-Attribute":
							start = int(line.split("\t")[1].split(" ")[1])
							end = int(line.split("\t")[1].split(" ")[2])
							text = line.split("\t")[-1]
							attribute_id = brat_id2attribute_id[brat_id]
							spatial_annotation[chat_id]["attributes"][attribute_id]["start"] = start
							spatial_annotation[chat_id]["attributes"][attribute_id]["end"] = end
							spatial_annotation[chat_id]["attributes"][attribute_id]["text"] = text
							spatial_annotation[chat_id]["attributes"][attribute_id]["subjects"] = []
							spatial_annotation[chat_id]["attributes"][attribute_id]["modifiers"] = []
							spatial_annotation[chat_id]["attributes"][attribute_id]["tags"] = set()

							# check basic attributes
							if brat_id in explicit_attributes:
								for explicit_attribute in explicit_attributes[brat_id]:
									if explicit_attribute in ["Ambiguous-Subj"]:
										spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("ambiguous")
									if explicit_attribute in ["Unannotatable", "No-Subj-Markable"]:
										spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("unannotatable")
							
							# add subjects
							if brat_id in explicit_subjects:
								for explicit_subject in explicit_subjects[brat_id]:
									if explicit_subject in brat_id2markable_id:
										spatial_annotation[chat_id]["attributes"][attribute_id]["subjects"].append(brat_id2markable_id[explicit_subject])
									else:
										spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].append("unannotatable")
							else:
								subject_markable_id = None
								for markable in markable_annotation[chat_id]["markables"]:
									if markable["generic"] or markable["predicative"]:
										continue
									if markable["start"] <= start:
										subject_markable_id = markable["markable_id"]
									else:
										break
								if subject_markable_id:
									spatial_annotation[chat_id]["attributes"][attribute_id]["subjects"].append(subject_markable_id)

							# add explicit modifiers
							if brat_id in explicit_modifiers:
								for explicit_modifier in explicit_modifiers[brat_id]:
									modifier = brat_id2modifiers[explicit_modifier]
									if explicit_modifier in explicit_attributes:
										canonical_function = explicit_attributes[explicit_modifier][0]
										if canonical_function == "Mod-Subtlty":
											canonical_function = "Mod-Subtlety"
									else:
										canonical_function = detect_canonical_function(modifier["text"])

									modifier["canonical-function"] = canonical_function

									spatial_annotation[chat_id]["attributes"][attribute_id]["modifiers"].append(modifier)

							# add explicit modifiers
							if brat_id in explicit_modifiers:
								for explicit_modifier in explicit_modifiers[brat_id]:
									spatial_annotation[chat_id]["attributes"][attribute_id]["modifiers"].append(brat_id2modifiers[explicit_modifier])

							# add annotator_note
							if brat_id in brat_id2annotator_note:
								spatial_annotation[chat_id]["attributes"][attribute_id]["paraphrase"] = brat_id2annotator_note[brat_id]
							
							# change from set to list
							spatial_annotation[chat_id]["attributes"][attribute_id]["tags"] = list(spatial_annotation[chat_id]["attributes"][attribute_id]["tags"])


			# Step 2. Add implicit modifiers
			with open(filename, "r") as fin:
				ann = fin.read()
				for line in ann.split("\n"):
					brat_id = line.split("\t")[0]
					if brat_id.startswith("T"):
						label = line.split("\t")[1].split(" ")[0]
						if label == "Modifier":
							modifier_start = int(line.split("\t")[1].split(" ")[1])
							modifier_end = int(line.split("\t")[1].split(" ")[2])
							# if not explicit modifier
							if brat_id not in explicit_modifiers.values():
								modifier = brat_id2modifiers[brat_id]
								if brat_id in explicit_attributes:
									canonical_function = explicit_attributes[brat_id][0]
									if canonical_function == "Mod-Subtlty":
										canonical_function = "Mod-Subtlety"
								else:
									canonical_function = detect_canonical_function(modifier["text"])
								modifier["canonical-function"] = canonical_function

								modified_brat_id = None
								for relation_or_attribute_brat_id, relation_or_attribute_end in sorted(brat_id2end.items(), key=lambda item: item[1]):
									if modifier_end < relation_or_attribute_end:
										modified_brat_id = relation_or_attribute_brat_id
										break
								if modified_brat_id:
									if modified_brat_id in brat_id2relation_id:
										relation_id = brat_id2relation_id[modified_brat_id]
										spatial_annotation[chat_id]["relations"][relation_id]["modifiers"].append(brat_id2modifiers[brat_id])
									elif modified_brat_id in brat_id2attribute_id:
										attribute_id = brat_id2attribute_id[modified_brat_id]
										spatial_annotation[chat_id]["attributes"][attribute_id]["modifiers"].append(brat_id2modifiers[brat_id])

			# Step 3. Add canonical relations
			with open(filename, "r") as fin:
				ann = fin.read()
				for line in ann.split("\n"):
					brat_id = line.split("\t")[0]
					if brat_id.startswith("T"):
						label = line.split("\t")[1].split(" ")[0]
						if label in ["Spatial-Relation", "Spatial-Relation-Markable"]:
							if brat_id in explicit_splits:
								continue

							start = int(line.split("\t")[1].split(" ")[1])
							end = int(line.split("\t")[1].split(" ")[2])
							text = line.split("\t")[-1]
							relation_id = brat_id2relation_id[brat_id]

							no_object = "no_object" in spatial_annotation[chat_id]["relations"][relation_id]["tags"]

							# add canonical relations
							if "unannotatable" not in spatial_annotation[chat_id]["relations"][relation_id]["tags"] and \
								"canonical_undefined" not in spatial_annotation[chat_id]["relations"][relation_id]["tags"]:
								subject_markable_texts = []
								for subject_markable_id in spatial_annotation[chat_id]["relations"][relation_id]["subjects"]:
									for markable in markable_annotation[chat_id]["markables"]:
										if markable["markable_id"] == subject_markable_id:
											subject_markable_texts.append(markable["text"])

								object_markable_texts = []
								for object_markable_id in spatial_annotation[chat_id]["relations"][relation_id]["objects"]:
									for markable in markable_annotation[chat_id]["markables"]:
										if markable["markable_id"] == object_markable_id:
											object_markable_texts.append(markable["text"])

								spatial_annotation[chat_id]["relations"][relation_id]["canonical-relations"] = \
									detect_canonical_relations(spatial_annotation[chat_id]["relations"][relation_id]["text"], 
															   subject_markable_texts,
															   object_markable_texts,
															   no_object,
															   spatial_annotation[chat_id]["relations"][relation_id]["splits"],
															   spatial_annotation[chat_id]["relations"][relation_id]["paraphrase"])

							if "unannotatable" not in spatial_annotation[chat_id]["relations"][relation_id]["tags"] and \
								len(spatial_annotation[chat_id]["relations"][relation_id]["canonical-relations"]) == 0:
								spatial_annotation[chat_id]["relations"][relation_id]["canonical-relations"] = ["undefined"]

	spatial_annotation = add_tags(args, spatial_annotation, dialogue_corpus)

	if args.annotator:
		dump_json(spatial_annotation, args.annotator + ".json")
	else:
		dump_json(spatial_annotation, "spatial_annotation.json")
		dump_json(chat_id2batch_id, "chat_id2batch_id.json")

def add_tags(args, spatial_annotation, dialogue_corpus):
	"""
		Add tags for spatial expressions, e.g.
			- intra_utterance_arguments
			- inter_utterance_subjs
			- inter_utterance_objs
			- multiple_subjs
			- multiple_objs
			- no_objs
			- well_ordered
			- ill_ordered
			- modified_weak
			- modified_strong
			- negated	
			- same_speaker
		Obsolete:
			- include_rare_words
			- ignorable_objects
			- unignorable_objects
			- common_words
			- rare_words
			- include_pronouns
	"""
	markable_annotation = read_json("markable_annotation.json")
	aggregated_referent_annotation = read_json("aggregated_referent_annotation.json")
	chat_ids = list(spatial_annotation.keys())

	vocab = Counter()
	for chat_id in markable_annotation.keys():
		for utterance in markable_annotation[chat_id]["text"].split('\n'):
			if len(utterance) > 0:
				vocab.update(word_tokenize(utterance[3:].lower()))
	# define rare words
	rare_words = [word for word in vocab.keys() if vocab[word] < 50]

	for chat_id in chat_ids:
		for dialogue_data in dialogue_corpus:
			if dialogue_data["uuid"] == chat_id:
				agent_ent_id2ent = {}
				for agent in [0, 1]:
					agent_ent_id2ent[agent] = {}
					for ent in dialogue_data["scenario"]["kbs"][agent]:
						agent_ent_id2ent[agent][ent["id"]] = ent
				break
		else:
			raise ValueError("chat_id {} not found!".format(chat_id))
			continue

		text = markable_annotation[chat_id]["text"]

		# create markable_id2utterance_id, expression_id2utterance_id
		markable_id2utterance_id = {}
		relation_id2utterance_id = {}
		attribute_id2utterance_id = {}
		modifier_id2utterance_id = {}
		markable_id2markable = {}

		markable_id2speaker = {}
		relation_id2speaker = {}
		attribute_id2speaker = {}

		current_utterance_len = 0
		for utterance_id, utterance in enumerate(text.split('\n')):
			current_utterance_len += len(utterance) + 1
			speaker = int(utterance[0])
			assert speaker in [0, 1]

			for markable in markable_annotation[chat_id]["markables"]:
				if markable["end"] < current_utterance_len and markable["markable_id"] not in markable_id2utterance_id:
					markable_id2utterance_id[markable["markable_id"]] = utterance_id
					markable_id2markable[markable["markable_id"]] = markable
					markable_id2speaker[markable["markable_id"]] = speaker

			for attribute_id, attribute in enumerate(spatial_annotation[chat_id]["attributes"]):
				if attribute["end"] < current_utterance_len and attribute_id not in attribute_id2utterance_id:
					attribute_id2utterance_id[attribute_id] = utterance_id
					attribute_id2speaker[attribute_id] = speaker

			for relation_id, relation in enumerate(spatial_annotation[chat_id]["relations"]):
				if relation["end"] < current_utterance_len and relation_id not in relation_id2utterance_id:
					relation_id2utterance_id[relation_id] = utterance_id
					relation_id2speaker[relation_id] = speaker

		# compute statistics for spatial relations
		for attribute_id, attribute in enumerate(spatial_annotation[chat_id]["attributes"]):
			# change from list --> set
			spatial_annotation[chat_id]["attributes"][attribute_id]["tags"] = set(spatial_annotation[chat_id]["attributes"][attribute_id]["tags"])

			intra_utterance_subjects = True
			same_speaker = True
			for markable_id in attribute["subjects"]:
				if markable_id2utterance_id[markable_id] != attribute_id2utterance_id[attribute_id]:
					intra_utterance_subjects = False
				if markable_id2speaker[markable_id] != attribute_id2speaker[attribute_id]:
					same_speaker = False
			
			if not intra_utterance_subjects:
				spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("inter_utterance_subjects")
			else:
				spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("intra_utterance_arguments")

			if same_speaker:
				spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("same_speaker")

			subj_start = -1
			for markable_id in attribute["subjects"]:
				subj_start = max(subj_start, markable_id2markable[markable_id]["start"])

			markable_start = []
			for markable_id in markable_id2markable.keys():
				markable_start.append(markable_id2markable[markable_id]["start"])
			markable_start = sorted(markable_start)

			well_ordered = subj_start <= attribute["start"]

			if well_ordered:
				spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("well_ordered")
			else:
				spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("ill_ordered")

			if len(attribute["modifiers"]) > 0:
				spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modified")
				for modifier in attribute["modifiers"]:
					if modifier["canonical-function"] in ["Mod-Extremity", "Mod-Certainty-Exactness"]:
						spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_strong")
					elif modifier["canonical-function"] in ["Mod-Subtlety", "Mod-Uncertainty-Approximation"]:
						spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_weak")
					elif modifier["canonical-function"] in ["Mod-Negation"]:
						spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_negated")
					else:
						spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_neutral")
			else:
				spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_neutral") # fix later
			"""
			if len(attribute["modifiers"]) > 0:
				spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modified")
				for modifier in attribute["modifiers"]:
					if modifier["canonical-function"] == "Mod-Extremity":
						spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_extremity")
					elif modifier["canonical-function"] == "Mod-Certainty-Exactness":
						spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_certainty")
					elif modifier["canonical-function"] == "Mod-Subtlety":
						spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_subtlety")
					elif modifier["canonical-function"] == "Mod-Uncertainty-Approximation":
						spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_uncertainty")
					elif modifier["canonical-function"] in ["Mod-Negation"]:
						spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_negated")
					else:
						spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_neutral")
			else:
				spatial_annotation[chat_id]["attributes"][attribute_id]["tags"].add("modification_none")
			"""
			# change from set --> list
			spatial_annotation[chat_id]["attributes"][attribute_id]["tags"] = list(spatial_annotation[chat_id]["attributes"][attribute_id]["tags"])

		# compute statistics for spatial relations
		for relation_id, relation in enumerate(spatial_annotation[chat_id]["relations"]):
			# change from list --> set
			spatial_annotation[chat_id]["relations"][relation_id]["tags"] = set(spatial_annotation[chat_id]["relations"][relation_id]["tags"])

			intra_utterance_arguments = True
			for markable_id in relation["subjects"] + relation["objects"]:
				if markable_id2utterance_id[markable_id] != relation_id2utterance_id[relation_id]:
					intra_utterance_arguments = False
					break

			intra_utterance_subjects = True
			for markable_id in relation["subjects"]:
				if markable_id2utterance_id[markable_id] != relation_id2utterance_id[relation_id]:
					intra_utterance_subjects = False
					break

			intra_utterance_objects = True
			for markable_id in relation["objects"]:
				if markable_id2utterance_id[markable_id] != relation_id2utterance_id[relation_id]:
					intra_utterance_objects = False
					break

			if intra_utterance_arguments:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("intra_utterance_arguments")
			else:
				if not intra_utterance_subjects:
					spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("inter_utterance_subjects")
				if not intra_utterance_objects:
					spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("inter_utterance_objects")

			same_speaker = True
			for markable_id in relation["subjects"] + relation["objects"]:
				if markable_id2speaker[markable_id] != relation_id2speaker[relation_id]:
					same_speaker = False
					break
			if same_speaker:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("same_speaker")


			subj_start = -1
			subj_end = -1
			for markable_id in relation["subjects"]:
				subj_start = max(subj_start, markable_id2markable[markable_id]["start"])
				subj_end = max(subj_end, markable_id2markable[markable_id]["end"])

			obj_start = float("inf")
			obj_end = float("inf")
			for markable_id in relation["objects"]:
				obj_start = min(obj_start, markable_id2markable[markable_id]["start"])
				obj_end = min(obj_end, markable_id2markable[markable_id]["end"])

			markable_start = []
			for markable_id in markable_id2markable.keys():
				markable_start.append(markable_id2markable[markable_id]["start"])
			markable_start = sorted(markable_start)

			if relation["start"] < subj_start:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("before_subj")
			else:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("after_subj")

			well_ordered = subj_start <= relation["start"] and relation["start"] <= obj_start

			if well_ordered:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("well_ordered")
			else:
				if obj_start < subj_start:
					spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("opposite_ordered")
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("ill_ordered")

			if len(relation["modifiers"]) > 0:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modified")
				for modifier in relation["modifiers"]:
					if modifier["canonical-function"] in ["Mod-Extremity", "Mod-Certainty-Exactness"]:
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_strong")
					elif modifier["canonical-function"] in ["Mod-Subtlety", "Mod-Uncertainty-Approximation"]:
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_weak")
					elif modifier["canonical-function"] in ["Mod-Negation"]:
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_negated")
					else:
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_neutral")
			else:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_neutral")
			"""
			if len(relation["modifiers"]) > 0:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modified")
				for modifier in relation["modifiers"]:
					if modifier["canonical-function"] == "Mod-Extremity":
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_extremity")
					elif modifier["canonical-function"] == "Mod-Certainty-Exactness":
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_certainty")
					elif modifier["canonical-function"] == "Mod-Subtlety":
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_subtlety")
					elif modifier["canonical-function"] == "Mod-Uncertainty-Approximation":
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_uncertainty")
					elif modifier["canonical-function"] in ["Mod-Negation"]:
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_negated")
					else:
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_neutral")
			else:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("modification_none") # fix later
			"""

			subj_tokens = []
			for markable_id in relation["subjects"]:
				subj_tokens.append(word_tokenize(markable_id2markable[markable_id]["text"].lower()))

			obj_tokens = []
			for markable_id in relation["objects"]:
				obj_tokens.append(word_tokenize(markable_id2markable[markable_id]["text"].lower()))

			include_rare_word = False
			for word in word_tokenize(relation["text"].lower()) + subj_tokens + obj_tokens:
				if word in rare_words:
					include_rare_word = True

			if not include_rare_word:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("no_rare_word")
			else:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("include_rare_word")

			include_pronoun = False
			pronoun_list = ["it", "that", "thats", "this", "its", "they", "their", "itself", "them", "those", "it's"]
			for markable_id in relation["subjects"] + relation["objects"]:
				if markable_id2markable[markable_id]["text"].lower() in pronoun_list:
					include_pronoun = True
					break

			if include_pronoun:
				spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("include_pronoun")

			test_relations = {}
			test_relations["direction"] = ["left", "right", "above", "below", "horizontal", "vertical", "diagonal"]
			test_relations["proximity"] = ["near", "far", "cluster", "alone"]
			test_relations["region"] = ["middle"]
			test_relations["color"] = ["lighter", "lightest", "darker", "darkest", "same color", "different color"]
			test_relations["size"] = ["smaller", "smallest", "larger", "largest", "same size", "different size"]

			speakers = set()

			subject_ent_ids = set()
			for subject_markable_id in relation["subjects"]:
				if subject_markable_id not in aggregated_referent_annotation[chat_id] or \
					('unidentifiable' in aggregated_referent_annotation[chat_id][subject_markable_id] and \
					aggregated_referent_annotation[chat_id][subject_markable_id]['unidentifiable']):
					continue
				for markable in markable_annotation[chat_id]["markables"]:
					if markable["markable_id"] == subject_markable_id:
						speakers.add(markable["speaker"])
				for referent in aggregated_referent_annotation[chat_id][subject_markable_id]["referents"]:
					_, speaker, ent_id = referent.split('_')
					speaker = int(speaker)
					for ent in dialogue_data['scenario']['kbs'][speaker]:
						if ent['id'] == ent_id:
							subject_ent_ids.add(ent['id'])
			
			object_ent_ids = set()
			for object_markable_id in relation["objects"]:
				if object_markable_id not in aggregated_referent_annotation[chat_id] or \
					('unidentifiable' in aggregated_referent_annotation[chat_id][object_markable_id] and \
					aggregated_referent_annotation[chat_id][object_markable_id]['unidentifiable']):
					continue
				for markable in markable_annotation[chat_id]["markables"]:
					if markable["markable_id"] == object_markable_id:
						speakers.add(markable["speaker"])
				for referent in aggregated_referent_annotation[chat_id][object_markable_id]["referents"]:
					_, speaker, ent_id = referent.split('_')
					speaker = int(speaker)
					for ent in dialogue_data['scenario']['kbs'][speaker]:
						if ent['id'] == ent_id:
							object_ent_ids.add(ent['id'])

			skip = "is_split" in relation["tags"]  or \
				"unannotatable" in relation["tags"] or \
				"canonical_undefined" in relation["tags"] or \
				"same_speaker" not in relation["tags"] or \
				len(speakers) == 0

			if not skip and len(relation["canonical-relations"]) > 0:
				agent = list(speakers)[0]

				subjects = {subject_ent_id:agent_ent_id2ent[agent][subject_ent_id] for subject_ent_id in subject_ent_ids}
				objects = {object_ent_id:agent_ent_id2ent[agent][object_ent_id] for object_ent_id in object_ent_ids}
				entities = agent_ent_id2ent[agent]

				text = relation["text"]
				tokens = word_tokenize(text)

				no_object = "no_object" in relation["tags"]
				negated = "modification_negated" in relation["tags"]

				if not no_object and not negated:
					ignorable_object = True
					for test_relation_category in test_relations.keys():
						for test_relation_type in test_relations[test_relation_category]:
							if test_relation_type in relation["canonical-relations"]:
								valid, satisfy, value = check_canonical_relation(test_relation_type, subjects, {}, entities, no_object=True)
								if valid and not satisfy:
									ignorable_object = False

					if ignorable_object:
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("ignorable_object")
					else:
						spatial_annotation[chat_id]["relations"][relation_id]["tags"].add("unignorable_object")

			# change from set --> list
			spatial_annotation[chat_id]["relations"][relation_id]["tags"] = list(spatial_annotation[chat_id]["relations"][relation_id]["tags"])

	return spatial_annotation

def compute_basic_statistics(args, dialogue_corpus):
	markable_annotation = read_json("markable_annotation.json")
	spatial_annotation = read_json("spatial_annotation.json")
	chat_id2batch_id = read_json("chat_id2batch_id.json")
	chat_ids = list(spatial_annotation.keys())

	attribute_statistics = {}
	relation_statistics = {}
	modifier_statistics = {}

	# count tags
	attribute_statistics["tags"] = {}
	attribute_statistics["tags"]["counter"] = Counter()
	relation_statistics["tags"] = {}
	relation_statistics["tags"]["counter"] = Counter()
	tag2occurence = {}

	# count expressions
	attribute_statistics["text"] = Counter()
	relation_statistics["text"] = Counter()
	modifier_statistics["text"] = Counter()

	relation_statistics["canonical-relations"] = {}
	relation_statistics["canonical-relations"]["counter"] = Counter()
	relation_statistics["canonical-relations"]["text"] = defaultdict(set)

	canonical_function_counter = defaultdict(Counter)
	canonical_relation_counter = defaultdict(Counter)
	unannotatable_attribute_counter = Counter()
	unannotatable_relation_counter = Counter()

	utterance_templates = Counter()

	for chat_id in chat_ids:
		batch_id = chat_id2batch_id[chat_id]

		# compute statistics for spatial relations
		for attribute_id, attribute in enumerate(spatial_annotation[chat_id]["attributes"]):
			attribute_statistics["text"][attribute["text"].lower()] += 1

			for tag in attribute["tags"]:
				attribute_statistics["tags"]["counter"][tag] += 1
				if tag == "unannotatable":
					unannotatable_attribute_counter[attribute["text"].lower()] += 1

			for modifier in attribute["modifiers"]:
				modifier_statistics["text"][modifier["text"].lower()] += 1
				canonical_function = modifier["canonical-function"]
				canonical_function_counter[canonical_function][modifier["text"].lower()] += 1

		# compute statistics for spatial relations
		for relation_id, relation in enumerate(spatial_annotation[chat_id]["relations"]):
			relation_statistics["text"][relation["text"]] += 1

			for tag in relation["tags"]:
				relation_statistics["tags"]["counter"][tag] += 1
				if tag not in tag2occurence:
					tag2occurence[tag] = {}
				if batch_id not in tag2occurence[tag]:
					tag2occurence[tag][batch_id] = {}
				if chat_id not in tag2occurence[tag][batch_id]:
					tag2occurence[tag][batch_id][chat_id] = []
				tag2occurence[tag][batch_id][chat_id].append(relation_id)
				if tag == "unannotatable":
					unannotatable_relation_counter[relation["text"].lower()] += 1

			for detected_canonical_relation in relation["canonical-relations"]:
				canonical_relation_counter[detected_canonical_relation][relation["text"].lower()] += 1

			for modifier in relation["modifiers"]:
				modifier_statistics["text"][modifier["text"].lower()] += 1
				canonical_function = modifier["canonical-function"]
				canonical_function_counter[canonical_function][modifier["text"].lower()] += 1

	total_attributes = sum(attribute_statistics["text"].values())
	print("total attributes: {}".format(total_attributes))
	print("average attributes per chat: {:.3f}".format(total_attributes / len(chat_ids)))
	for tag in attribute_statistics["tags"]["counter"].keys():
		count = attribute_statistics["tags"]["counter"][tag]
		print("{} rate: {:.3f}%".format(tag, 100.0 * count / total_attributes))
	print("unique: {}".format(len(attribute_statistics["text"].keys())))

	print("")

	total_relations = sum(relation_statistics["text"].values())
	print("total relations: {}".format(total_relations))
	print("average relations per chat: {:.3f}".format(total_relations / len(chat_ids)))
	for tag in relation_statistics["tags"]["counter"].keys():
		count = relation_statistics["tags"]["counter"][tag]
		print("{} rate: {:.3f}%".format(tag, 100.0 * count / total_relations))
	print("unique: {}".format(len(relation_statistics["text"].keys())))

	print("")

	for tag in args.print_tags:
		if tag not in relation_statistics["tags"]["counter"]:
			raise ValueError('{} tag not found'.format(tag))
		else:
			for batch_id in tag2occurence[tag]:
				for chat_id in tag2occurence[tag][batch_id]:
					print(batch_id, chat_id, tag2occurence[tag][batch_id][chat_id])

	pdb.set_trace()

def test_canonical_relations(args, dialogue_corpus, referent_annotation):
	markable_annotation = read_json("markable_annotation.json")
	aggregated_referent_annotation = read_json("aggregated_referent_annotation.json")
	spatial_annotation = read_json("spatial_annotation.json")
	chat_id2batch_id = read_json("chat_id2batch_id.json")

	chat_ids = list(spatial_annotation.keys())

	test_relations = {}
	test_relations["direction"] = ["left", "right", "above", "below", "horizontal", "vertical", "diagonal"]
	test_relations["proximity"] = ["near", "far", "alone"]
	test_relations["region"] = ["interior", "exterior"]
	test_relations["color"] = ["lighter", "lightest", "darker", "darkest", "same color", "different color"]
	test_relations["size"] = ["smaller", "smallest", "larger", "largest", "same size", "different size"]

	test_results = {}
	for test_relation_category in test_relations.keys():
		test_results[test_relation_category] = {}
		for test_relation_type in test_relations[test_relation_category]:
			test_results[test_relation_category][test_relation_type] = {}
			test_results[test_relation_category][test_relation_type]["all"] = [] # (chat_id, relation_id)
			test_results[test_relation_category][test_relation_type]["satisfy"] = []
			test_results[test_relation_category][test_relation_type]["unsatisfy"] = []
			test_results[test_relation_category][test_relation_type]["invalid"] = []
			test_results[test_relation_category][test_relation_type]["values"] = []

	tag_results = {}
	tag_results["__any__"] = {}
	tag_results["__any__"]["all"] = [] # (chat_id, relation_id)
	tag_results["__any__"]["satisfy"] = []
	tag_results["__any__"]["unsatisfy"] = []
	tag_results["__any__"]["invalid"] = []
	tag_results["__any__"]["values"] = []

	modified_results = {}
	for modification_type in ["modification_neutral", "modification_weak", "modification_strong"]:
		modified_results[modification_type] = {}
		for value_type in ["xy_value", "coef_value", "distance_value", "color_value", "size_value"]:
			modified_results[modification_type][value_type] = []

	for chat_id in chat_ids:
		for dialogue_data in dialogue_corpus:
			if dialogue_data["uuid"] == chat_id:
				agent_ent_id2ent = {}
				for agent in [0, 1]:
					agent_ent_id2ent[agent] = {}
					for ent in dialogue_data["scenario"]["kbs"][agent]:
						agent_ent_id2ent[agent][ent["id"]] = ent
				break
		else:
			raise ValueError("chat_id {} not found!".format(chat_id))
			continue

		for relation_id, relation in enumerate(spatial_annotation[chat_id]["relations"]):
			speakers = set()

			subject_ent_ids = set()
			for subject_markable_id in relation["subjects"]:
				if subject_markable_id not in aggregated_referent_annotation[chat_id] or \
					('unidentifiable' in aggregated_referent_annotation[chat_id][subject_markable_id] and \
					aggregated_referent_annotation[chat_id][subject_markable_id]['unidentifiable']):
					continue
				for markable in markable_annotation[chat_id]["markables"]:
					if markable["markable_id"] == subject_markable_id:
						speakers.add(markable["speaker"])
				for referent in referent_annotation[chat_id][subject_markable_id]["referents"]:
					_, speaker, ent_id = referent.split('_')
					speaker = int(speaker)
					for ent in dialogue_data['scenario']['kbs'][speaker]:
						if ent['id'] == ent_id:
							subject_ent_ids.add(ent['id'])
			
			object_ent_ids = set()
			for object_markable_id in relation["objects"]:
				if object_markable_id not in referent_annotation[chat_id] or \
					('unidentifiable' in aggregated_referent_annotation[chat_id][object_markable_id] and \
					aggregated_referent_annotation[chat_id][object_markable_id]['unidentifiable']):
					continue
				for markable in markable_annotation[chat_id]["markables"]:
					if markable["markable_id"] == object_markable_id:
						speakers.add(markable["speaker"])
				for referent in referent_annotation[chat_id][object_markable_id]["referents"]:
					_, speaker, ent_id = referent.split('_')
					speaker = int(speaker)
					for ent in dialogue_data['scenario']['kbs'][speaker]:
						if ent['id'] == ent_id:
							object_ent_ids.add(ent['id'])

			if "is_split" in relation["tags"]  or \
				"unannotatable" in relation["tags"] or \
				"canonical_undefined" in relation["tags"] or \
				"same_speaker" not in relation["tags"] or \
				len(speakers) == 0:
				continue

			assert len(speakers) == 1

			agent = list(speakers)[0]

			subjects = {subject_ent_id:agent_ent_id2ent[agent][subject_ent_id] for subject_ent_id in subject_ent_ids}
			objects = {object_ent_id:agent_ent_id2ent[agent][object_ent_id] for object_ent_id in object_ent_ids}
			entities = agent_ent_id2ent[agent]

			text = relation["text"]
			tokens = word_tokenize(text)

			no_object = "no_object" in relation["tags"]
			negated = "modification_negated" in relation["tags"]

			for test_relation_category in test_relations.keys():
				for test_relation_type in test_relations[test_relation_category]:
					if test_relation_type in relation["canonical-relations"]:
						valid, satisfy, value = check_canonical_relation(test_relation_type, subjects, objects, entities, no_object)
						if negated:
							continue

						if valid:
							if satisfy:
								test_results[test_relation_category][test_relation_type]["satisfy"].append((chat_id2batch_id[chat_id], chat_id, relation_id))
								tag_results["__any__"]["satisfy"].append((chat_id2batch_id[chat_id], chat_id, relation_id))
							else:
								test_results[test_relation_category][test_relation_type]["unsatisfy"].append((chat_id2batch_id[chat_id], chat_id, relation_id))
								tag_results["__any__"]["unsatisfy"].append((chat_id2batch_id[chat_id], chat_id, relation_id))
						else:
							test_results[test_relation_category][test_relation_type]["invalid"].append((chat_id2batch_id[chat_id], chat_id, relation_id))
							tag_results["__any__"]["invalid"].append((chat_id2batch_id[chat_id], chat_id, relation_id))
						if value is not None:
							test_results[test_relation_category][test_relation_type]["values"].append(value)
						test_results[test_relation_category][test_relation_type]["all"].append((chat_id2batch_id[chat_id], chat_id, relation_id))
						tag_results["__any__"]["all"].append((chat_id2batch_id[chat_id], chat_id, relation_id))

						# get tag reslults
						for tag in relation["tags"]:
							if tag not in tag_results:
								# initialize
								tag_results[tag] = {}
								tag_results[tag]["all"] = [] # (chat_id, relation_id)
								tag_results[tag]["satisfy"] = []
								tag_results[tag]["unsatisfy"] = []
								tag_results[tag]["invalid"] = []
								tag_results[tag]["values"] = []
							if valid:
								if satisfy:
									tag_results[tag]["satisfy"].append((chat_id2batch_id[chat_id], chat_id, relation_id))
								else:
									tag_results[tag]["unsatisfy"].append((chat_id2batch_id[chat_id], chat_id, relation_id))
							else:
								tag_results[tag]["invalid"].append((chat_id2batch_id[chat_id], chat_id, relation_id))
							
							if tag.startswith("modification_"):
								if value is not None:
									if test_relation_type in ["left", "right", "above", "below"]:
										modified_results[tag]["xy_value"].append(abs(value))
									if test_relation_type in ["lighter", "darker"]:
										modified_results[tag]["color_value"].append(abs(value))
									if test_relation_type in ["smaller", "larger"]:
										modified_results[tag]["size_value"].append(abs(value))
							
							tag_results[tag]["all"].append((chat_id2batch_id[chat_id], chat_id, relation_id))

	satisfy_results = []
	valid_results = []
	count_results = []

	# print final results
	for test_relation_category in test_relations.keys():
		print(test_relation_category)
		overall_total = 0
		overall_satisfy = 0
		overall_unsatisfy = 0
		overall_invalid = 0
		for test_relation_type in test_relations[test_relation_category]:
			total = len(test_results[test_relation_category][test_relation_type]["all"])
			satisfy = len(test_results[test_relation_category][test_relation_type]["satisfy"])
			unsatisfy = len(test_results[test_relation_category][test_relation_type]["unsatisfy"])
			invalid = len(test_results[test_relation_category][test_relation_type]["invalid"])
			if len(test_results[test_relation_category][test_relation_type]["values"]) > 0:
				average_value = np.mean(test_results[test_relation_category][test_relation_type]["values"])
			else:
				average_value = 0
			print("    {}: satisfy/valid {:.2f}/{:.2f} %, invalid {:.2f} % (out of {}), average value {:.2f}".format(test_relation_type,
																									100.0 * satisfy / total,
																									100.0 * (satisfy + unsatisfy) / total,
																									100.0 * invalid / total,
																									total,
																									average_value))
			if args.print_satisfy:
				print("    --- satisfy --")
				print("    " + str(test_results[test_relation_category][test_relation_type]["satisfy"]))
			if args.print_unsatisfy:
				print("    --- unsatisfy --")
				print("    " + str(test_results[test_relation_category][test_relation_type]["unsatisfy"]))
			if args.print_invalid:
				print("    --- invalid --")
				print("    " + str(test_results[test_relation_category][test_relation_type]["invalid"]))

			satisfy_results.append(round(1000.0 * satisfy / total) / 10)
			valid_results.append(round(1000.0 * (satisfy + unsatisfy) / total) / 10)
			count_results.append(total)
			overall_total += total
			overall_satisfy += satisfy
			overall_unsatisfy += unsatisfy
			overall_invalid += invalid

		if len(test_relations[test_relation_category]) > 1:
			satisfy_results.append(round(1000.0 * overall_satisfy / overall_total) / 10)
			valid_results.append(round(1000.0 * (overall_satisfy + overall_unsatisfy) / overall_total) / 10)
			count_results.append(overall_total)

	print("\ntag results")
	for tag in tag_results.keys():
		total = len(tag_results[tag]["all"])
		satisfy = len(tag_results[tag]["satisfy"])
		unsatisfy = len(tag_results[tag]["unsatisfy"])
		invalid = len(tag_results[tag]["invalid"])
		print("    {}: satisfy/valid {:.2f}/{:.2f} %, invalid {:.2f} % (out of {})".format(tag,
																									100.0 * satisfy / total,
																									100.0 * (satisfy + unsatisfy) / total,
																									100.0 * invalid / total,
																									total))		
		if args.print_satisfy:
			print("    --- satisfy --")
			print("    " + str(tag_results[tag]["satisfy"]))
		if args.print_unsatisfy:
			print("    --- unsatisfy --")
			print("    " + str(tag_results[tag]["unsatisfy"]))
		if args.print_invalid:
			print("    --- invalid --")
			print("    " + str(tag_results[tag]["invalid"]))

		if tag.startswith("modification_"):
			for value_type in ["xy_value", "color_value", "size_value"]:
				if len(modified_results[tag][value_type]) > 0:
					average_value = np.mean(modified_results[tag][value_type])
					print("        {}: {:.2f} (out of {})".format(value_type, average_value, len(modified_results[tag][value_type])))
			

	return satisfy_results, valid_results, count_results

def test_all_models(args, dialogue_corpus):
	latex_results = []

	referent_annotation = read_json("ref_model_referent_annotation.json")
	satisfy_results, valid_results, count_results = test_canonical_relations(args, dialogue_corpus, referent_annotation)
	latex_results.append(count_results)
	latex_results.append(satisfy_results)
	latex_results.append(valid_results)

	ablated_satisfy_results = []
	ablated_valid_results = []
	referent_annotation = read_json("ref_no_loc_model_referent_annotation.json")
	satisfy_results, valid_results, _ = test_canonical_relations(args, dialogue_corpus, referent_annotation)
	ablated_satisfy_results = satisfy_results[:15]
	ablated_valid_results = valid_results[:15]

	referent_annotation = read_json("ref_no_color_model_referent_annotation.json")
	satisfy_results, valid_results, _ = test_canonical_relations(args, dialogue_corpus, referent_annotation)
	ablated_satisfy_results += satisfy_results[15:22]
	ablated_valid_results += valid_results[15:22]

	referent_annotation = read_json("ref_no_size_model_referent_annotation.json")
	satisfy_results, valid_results, _ = test_canonical_relations(args, dialogue_corpus, referent_annotation)
	ablated_satisfy_results += satisfy_results[22:]
	ablated_valid_results += valid_results[22:]

	latex_results.append(ablated_satisfy_results)
	latex_results.append(ablated_valid_results)

	referent_annotation = read_json("num_ref_model_referent_annotation.json")
	satisfy_results, valid_results, _ = test_canonical_relations(args, dialogue_corpus, referent_annotation)
	latex_results.append(satisfy_results)
	latex_results.append(valid_results)

	ablated_satisfy_results = []
	ablated_valid_results = []
	referent_annotation = read_json("num_ref_no_loc_model_referent_annotation.json")
	satisfy_results, valid_results, _ = test_canonical_relations(args, dialogue_corpus, referent_annotation)
	ablated_satisfy_results = satisfy_results[:15]
	ablated_valid_results = valid_results[:15]

	referent_annotation = read_json("num_ref_no_color_model_referent_annotation.json")
	satisfy_results, valid_results, _ = test_canonical_relations(args, dialogue_corpus, referent_annotation)
	ablated_satisfy_results += satisfy_results[15:22]
	ablated_valid_results += valid_results[15:22]

	referent_annotation = read_json("num_ref_no_size_model_referent_annotation.json")
	satisfy_results, valid_results, _ = test_canonical_relations(args, dialogue_corpus, referent_annotation)
	ablated_satisfy_results += satisfy_results[22:]
	ablated_valid_results += valid_results[22:]

	latex_results.append(ablated_satisfy_results)
	latex_results.append(ablated_valid_results)

	referent_annotation = read_json("aggregated_referent_annotation.json")
	satisfy_results, valid_results, _ = test_canonical_relations(args, dialogue_corpus, referent_annotation)
	latex_results.append(satisfy_results)
	latex_results.append(valid_results)

	latex_results = np.array(latex_results).transpose()
	category = ["\\multirow{8}{*}{Direction}\\phantom{..}", "", "", "", "", "", "", ""]
	category += ["\\multirow{4}{*}{Proximity}", "", "", ""]
	category += ["\\multirow{3}{*}{Region}", "", ""]
	category += ["\\multirow{7}{*}{Color}", "", "", "", "", "", ""]
	category += ["\\multirow{7}{*}{Size}", "", "", "", "", "", ""]
	relation = ["\\textit{left}", "\\textit{right}", "\\textit{above}", "\\textit{below}", "\\textit{horizontal}", "\\textit{vertical}", "\\textit{diagonal}", "All"]
	relation += ["\\textit{near}", "\\textit{far}", "\\textit{alone}", "All"]
	relation += ["\\textit{interior}", "\\textit{exterior}", "All"]
	relation += ["\\textit{lighter}", "\\textit{lightest}", "\\textit{darker}", "\\textit{darkest}", "\\textit{same}", "\\textit{different}", "All"]
	relation += ["\\textit{smaller}", "\\textit{smallest}", "\\textit{larger}", "\\textit{largest}", "\\textit{same}", "\\textit{different}", "All"]

	print("")
	print("\\midrule")
	for i in range(latex_results.shape[0]):
		latex_row = []
		latex_row.append(category[i])
		latex_row.append(relation[i])
		latex_row.append(str(int(latex_results[i][0])))
		for j in range(1, latex_results.shape[1]):
			latex_row.append("{:.1f}".format(latex_results[i][j]))
		print(" & ".join(latex_row) + " \\\\")
		if i in [6, 10, 13, 20, 27]:
			print("\\intermidrule")
		if i in [7, 11, 14, 21]:
			print("\\midrule")

	print("\\bottomrule")
	print("")
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=1, help='random seed')
	parser.add_argument('--scenario_file', type=str, default="aaai_train_scenarios.json")
	parser.add_argument('--scenario_file_2', type=str, default="aaai_train_scenarios_2.json")
	parser.add_argument('--transcript_file', type=str, default="final_transcripts.json")

	parser.add_argument('--output_brat_format', action='store_true', default=False)
	parser.add_argument('--print_tags', nargs='*', type=str, default=[])
	parser.add_argument('--output_spatial_annotation', action='store_true', default=False)
	parser.add_argument('--compute_basic_statistics', action='store_true', default=False)
	parser.add_argument('--test_canonical_relations', action='store_true', default=False)
	parser.add_argument('--referent_annotation', type=str, default="aggregated_referent_annotation.json")
	parser.add_argument('--test_all_models', action='store_true', default=False)
	parser.add_argument('--print_satisfy', action='store_true', default=False)
	parser.add_argument('--print_unsatisfy', action='store_true', default=False)
	parser.add_argument('--print_invalid', action='store_true', default=False)
	parser.add_argument('--annotator', type=str, default=None)

	args = parser.parse_args()

	np.random.seed(args.seed)

	dialogue_corpus = read_json(args.transcript_file)
	scenario_list = read_json(args.scenario_file)
	scenario_list += read_json(args.scenario_file_2)

	if args.output_brat_format:
		output_brat_format(args, dialogue_corpus)

	if args.output_spatial_annotation:
		output_spatial_annotation(args, dialogue_corpus)

	if args.compute_basic_statistics:
		compute_basic_statistics(args, dialogue_corpus)

	if args.test_canonical_relations:
		referent_annotation = read_json(args.referent_annotation)
		test_canonical_relations(args, dialogue_corpus, referent_annotation)

	if args.test_all_models:
		test_all_models(args, dialogue_corpus)

