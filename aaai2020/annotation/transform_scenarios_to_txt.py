import argparse
import json
import os
import re

from nltk import word_tokenize
import pdb

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

def create_input(kb, args):
	input_tokens = []
	xs = []
	ys = []
	sizes = []
	colors = []
	for obj in kb:
		xs.append(obj['x'])
		ys.append(obj['y'])
		sizes.append(obj['size'])
		colors.append(int(re.search(r"[\d]+", obj['color']).group(0)))
		if args.normalize:
			xs[-1] = normalize_val(xs[-1], args.svg_radius + args.margin, args.svg_radius * 2)
			ys[-1] = normalize_val(ys[-1], args.svg_radius + args.margin, args.svg_radius * 2)
			sizes[-1] = normalize_val(sizes[-1], args.base_size, args.size_range)
			colors[-1] = normalize_val(colors[-1], args.base_color, args.color_range)
		if args.drop_x:
			xs[-1] = 0
		if args.drop_y:
			ys[-1] = 0
		if args.drop_size:
			sizes[-1] = 0
		if args.drop_color:
			colors[-1] = 0
		input_tokens += [str(xs[-1]), str(ys[-1]), str(sizes[-1]), str(colors[-1])]
	return input_tokens

def create_real_ids(kb):
	real_ids = []
	for obj in kb:
		real_ids.append(obj['id'])
	return real_ids

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', type=str, default='scenarios.json')
	parser.add_argument('--output_file', type=str, default='scenarios.txt')
	parser.add_argument('--color_range', type=int, default=150, help='range of color')
	parser.add_argument('--size_range', type=int, default=6, help='range of size')
	parser.add_argument('--base_size', type=int, default=10, help='base of size')
	parser.add_argument('--normalize', action='store_true', default=False)
	parser.add_argument('--seed', type=int, default=1, help='range of size')

	# drop attributes
	parser.add_argument('--drop_color', action='store_true', default=False)
	parser.add_argument('--drop_size', action='store_true', default=False)
	parser.add_argument('--drop_x', action='store_true', default=False)
	parser.add_argument('--drop_y', action='store_true', default=False)

	args = parser.parse_args()
	args.base_color = 128
	args.svg_radius = 200
	args.margin = 15

	scenarios = read_json(args.input_file)

	with open(args.output_file, 'w') as out:
		for scenario in scenarios:
			out.write(scenario['uuid'] + "\n")
			for agent in [0, 1]:
				out.write(" ".join(create_input(scenario['kbs'][agent], args)) + "\n")
			for agent in [0, 1]:
				out.write(" ".join(create_real_ids(scenario['kbs'][agent])) + "\n")
