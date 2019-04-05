import argparse
import json
import os
import re

from nltk import word_tokenize
import pdb
import numpy as np

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

def normalize_val(val, base_val, val_range):
	normalized_val = (val - base_val) / (val_range / 2)
	assert normalized_val >= -1 and normalized_val <= 1
	return normalized_val

def create_input(kb, args, partner=False):
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

def create_dialogue(events, agent):
	dialogue_tokens = []
	for event in events:
		if event['action'] == 'message':
			if agent == event['agent']:
				dialogue_tokens.append("YOU:")
			else:
				dialogue_tokens.append("THEM:")
			# TODO: more sophisticated tokenization
			dialogue_tokens += word_tokenize(event['data'].lower())
			dialogue_tokens.append("<eos>")
	return ['<dialogue>'] + dialogue_tokens + ['<selection>', '</dialogue>']

def create_output(kb, events, agent):
	ids = []
	for obj in kb:
		ids.append(obj['id'])
	select_id = None
	for event in events:
		if event['action'] == 'select' and agent == event['agent']:
			select_id = ids.index(event['data'])
	return ['<output>', str(select_id), '</output>']

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_name', type=str, default='data/final_transcripts.json')
	parser.add_argument('--color_range', type=int, default=150, help='range of color')
	parser.add_argument('--size_range', type=int, default=6, help='range of size')
	parser.add_argument('--base_size', type=int, default=10, help='range of size')
	parser.add_argument('--normalize', action='store_true', default=False)
	parser.add_argument('--valid_proportion', type=float, default=0.1)
	parser.add_argument('--test_proportion', type=float, default=0.1)
	parser.add_argument('--uncorrelated', action='store_true', default=False)
	parser.add_argument('--success_only', action='store_true', default=False)
	args = parser.parse_args()

	# current support
	args.base_color = 128
	args.svg_radius = 200
	args.margin = 15
	
	args.train_proportion = 1 - args.valid_proportion - args.test_proportion

	raw_corpus = read_json(args.file_name)
	total_size = len(raw_corpus)
	split_index = [0, int(total_size * args.train_proportion),
					int(total_size * (args.train_proportion + args.valid_proportion)), -1]
	tags = Tags()

	output_file_names = ['train', 'valid', 'test']
	if args.uncorrelated:
		output_file_names = [output_file_name + '_uncorrelated'
							for output_file_name in output_file_names]
	elif args.success_only:
		output_file_names = [output_file_name + '_success_only'
							for output_file_name in output_file_names]

	for i, output_file in enumerate(output_file_names):
		with open('{}.txt'.format(output_file), 'w') as out:
			start = split_index[i]
			end = split_index[i+1]
			for chat in raw_corpus[start:end]:
				for agent in [0,1]:
					if args.uncorrelated and agent == 1:
						continue
					elif args.success_only and chat['outcome']['reward'] == 0:
						continue
					tokens = []
					tokens += create_input(chat['scenario']['kbs'][agent], args)
					tokens += create_dialogue(chat['events'], agent)
					tokens += create_output(chat['scenario']['kbs'][agent], chat['events'], agent)
					out.write(" ".join(tokens) + "\n")
