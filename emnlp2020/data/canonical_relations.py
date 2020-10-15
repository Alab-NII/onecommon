import numpy as np
from sklearn.linear_model import LinearRegression
import pdb
from nltk import word_tokenize, pos_tag, bigrams, ngrams
from itertools import permutations, combinations 


# Dictionary of canonical relations
canonical_relations = {}

# Direction
canonical_relations["direction"] = {}
canonical_relations["direction"]["left"] = ["left", "west", "northwest", "southwest"]
canonical_relations["direction"]["right"] = ["right", "east", "northeast", "southeast"]
canonical_relations["direction"]["above"] = ["above", "up", "top", "high", "higher", "highest", "north", "northwest", "northeast",  "upper", "topmost"]
canonical_relations["direction"]["below"] = ["below", "down", "beneath", "bottom", "low", "lower", "lowest", "south", "southeast", "southwest", "underneath", "under"]
canonical_relations["direction"]["horizontal"] = ["horizontal", "horizontally"]
canonical_relations["direction"]["vertical"] = ["vertical", "vertically"]
canonical_relations["direction"]["diagonal"] = ["diagonal", "diagonally", "slanted", "slope", "sloping"]

# Region and Proximity
canonical_relations["proximity"] = {}
canonical_relations["proximity"]["near"] = ["near", ("close", "to"), ("next", "to"), "cluster", "clustered", "group", "grouped", "grouping", "together"]
canonical_relations["proximity"]["far"] = ["far", ("away", "from"), ("apart", "from")]
canonical_relations["proximity"]["alone"] = ["alone", "lonesome", "lone", "lonely", "isolated", "solo"]

canonical_relations["region"] = {}
canonical_relations["region"]["interior"] = ["interior", "middle", "center", "centered", "between", "inside"]
canonical_relations["region"]["exterior"] = ["exterior", "outside", "outer", "edge", "border"]

# Color comparison
canonical_relations["color"] = {}
canonical_relations["color"]["lighter"] = ["lighter"]
canonical_relations["color"]["lightest"] = ["lightest"]
canonical_relations["color"]["darker"] = ["darker"]
canonical_relations["color"]["darkest"] = ["darkest"]
canonical_relations["color"]["same color"] = [("same", "color"), ("same", "colors"), ("same", "colored"), ("similar", "color"), ("similar", "colors"), ("similar", "colored"), ("same", "shade"), ("same", "shades"), ("same", "shaded"), ("similar", "shade"), ("similar", "shades"), ("similar", "shaded"), ("similar", "in", "color"), ("identical", "looking")]
canonical_relations["color"]["different color"] = [("different", "color"), ("different", "colors"), ("different", "colored"), ("different", "shade"), ("different", "shades"), ("different", "shaded"), ("opposite", "color"), ("opposite", "colors"), ("opposite", "colored"), ("different", "in", "color"), ("different", "in", "colors"), ("opposite", "in", "color"), ("opposite", "in", "colors")]

# Size comparison
canonical_relations["size"] = {}
canonical_relations["size"]["smaller"] = ["smaller", "tinier"]
canonical_relations["size"]["smallest"] = ["smallest", "tiniest"]
canonical_relations["size"]["larger"] = ["larger", "bigger"]
canonical_relations["size"]["largest"] = ["largest", "biggest"]
canonical_relations["size"]["same size"] = [("same", "size"), ("same", "sized"), ("similar", "size"), ("similar", "sized"), ("similar", "sizes"), ("similar", "in", "size"), ("identical", "in", "size"), ("identical", "looking")]
canonical_relations["size"]["different size"] = [("different", "size"), ("different", "sized"), ("different", "sizes"), ("opposite", "in", "size"), ("opposite", "in", "sizes")]

#requires_object = ["middle", "near", "far"]
requires_object = ["far"]

def extract_color(rgb_color):
	return int(rgb_color.split(',')[1])

def compute_distance(loc_1, loc_2):
	return np.sqrt((loc_1[0] - loc_2[0]) ** 2 + (loc_1[1] - loc_2[1]) ** 2)

def detect_canonical_relations(relation_text, subject_markable_texts, object_markable_texts, no_object, split_texts, paraphrase):
	detected_canonical_relations = []

	# add relation text
	if len(paraphrase) > 0:
		text = paraphrase
		text = " ".join(text.split(";")) # annotator notes are split by ;
	else:
		text = relation_text
		text = " ".join(text.split("/")) # fix tokenization mistake
		text = " ".join(text.split("-")) # fix tokenization mistake
	tokens = word_tokenize(text.lower())

	# automatically detect canonical relations
	for canonical_relation_catergoy in canonical_relations.keys():
		for canonical_relation_type in canonical_relations[canonical_relation_catergoy]:
			if canonical_relation_type in requires_object and no_object:
				continue
			if isinstance(canonical_relations[canonical_relation_catergoy][canonical_relation_type], list):
				key_words = canonical_relations[canonical_relation_catergoy][canonical_relation_type]
				for key_word in key_words:
					if isinstance(key_word, tuple):
						if key_word in ngrams(tokens, len(key_word)):
							detected_canonical_relations.append(canonical_relation_type)
					else:
						if key_word in tokens:
							detected_canonical_relations.append(canonical_relation_type)
			else:
				# keyword must match exactly
				key_word = canonical_relations[canonical_relation_catergoy][canonical_relation_type]
				if key_word == text:
					detected_canonical_relations.append(canonical_relation_type)

	return detected_canonical_relations


def check_canonical_relation(relation_type, subjects, objects, entities, no_object):
	base_x = 215
	base_y = 215
	base_color = 128
	base_size = 10

	"""
		Check relations for orientation/direction.
	"""

	if not no_object and len(objects) == 0:
		return False, False, None

	if relation_type == "left":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_xs = []
		for subj in subjects.values():
			subj_xs.append(subj["x"])

		obj_xs = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_xs.append(obj["x"])

		if len(obj_xs) > 0:
			value = np.mean(subj_xs) - np.mean(obj_xs)
		else:
			value = np.mean(subj_xs) - base_x

		if value < 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "right":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_xs = []
		for subj in subjects.values():
			subj_xs.append(subj["x"])

		obj_xs = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_xs.append(obj["x"])

		if len(obj_xs) > 0:
			value = np.mean(subj_xs) - np.mean(obj_xs)
		else:
			value = np.mean(subj_xs) - base_x

		if value > 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "above":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_ys = []
		for subj in subjects.values():
			subj_ys.append(subj["y"])

		obj_ys = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_ys.append(obj["y"])

		if len(obj_ys) > 0:
			value = np.mean(subj_ys) - np.mean(obj_ys)
		else:
			value = np.mean(subj_ys) - base_y
		
		if value < 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "below":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_ys = []
		for subj in subjects.values():
			subj_ys.append(subj["y"])

		obj_ys = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_ys.append(obj["y"])

		if len(obj_ys) > 0:
			value = np.mean(subj_ys) - np.mean(obj_ys)
		else:
			value = np.mean(subj_ys) - base_y
		
		if value > 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "horizontal":
		xs = []
		ys = []

		for subj in subjects.values():
			xs.append(subj["x"])
			ys.append(subj["y"])

		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				xs.append(obj["x"])
				ys.append(obj["y"])

		if len(xs) > 1:
			valid = True
		else:
			return False, False, None

		xs = np.array(xs).reshape(-1, 1)
		ys = np.array(ys).reshape(-1, 1)

		reg = LinearRegression().fit(xs, ys)

		value = abs(reg.coef_[0][0])

		if value < 1 / 3:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "vertical":
		xs = []
		ys = []

		# Rotate the coordinates 90 degrees
		for subj in subjects.values():
			xs.append(-subj["y"])
			ys.append(subj["x"])

		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				xs.append(-obj["y"])
				ys.append(obj["x"])

		if len(xs) > 1:
			valid = True
		else:
			return False, False, None

		xs = np.array(xs).reshape(-1, 1)
		ys = np.array(ys).reshape(-1, 1)

		reg = LinearRegression().fit(xs, ys)

		value = abs(reg.coef_[0][0])

		if value < 1 / 3:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "diagonal":
		xs = []
		ys = []

		for subj in subjects.values():
			xs.append(subj["x"])
			ys.append(subj["y"])

		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				xs.append(obj["x"])
				ys.append(obj["y"])

		if len(xs) > 1:
			valid = True
		else:
			return False, False, None

		xs = np.array(xs).reshape(-1, 1)
		ys = np.array(ys).reshape(-1, 1)

		reg = LinearRegression().fit(xs, ys)

		value_horizontal = abs(reg.coef_[0][0])

		xs = []
		ys = []

		# Rotate the coordinates 90 degrees
		for subj in subjects.values():
			xs.append(-subj["y"])
			ys.append(subj["x"])

		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				xs.append(-obj["y"])
				ys.append(obj["x"])

		xs = np.array(xs).reshape(-1, 1)
		ys = np.array(ys).reshape(-1, 1)

		reg = LinearRegression().fit(xs, ys)

		value_vertical = abs(reg.coef_[0][0])

		value = min(value_vertical, value_horizontal)

		if value > 1 / 3:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	"""
		Check relations for proximity.
	"""
	if relation_type == "near":
		locs = []
		all_locs = []

		for subj in subjects.values():
			locs.append((subj["x"], subj["y"]))

		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				locs.append((obj["x"], obj["y"]))

		for ent in entities.values():
			all_locs.append((ent["x"], ent["y"]))

		if len(locs) > 1:
			valid = True
		else:
			return False, False, None

		distance = []
		all_distance = []
		for loc_1, loc_2 in combinations(locs, 2):
			distance.append(compute_distance(loc_1, loc_2))
		for loc_1, loc_2 in combinations(all_locs, 2):
			all_distance.append(compute_distance(loc_1, loc_2))

		value = np.mean(distance) / np.mean(all_distance)

		if value < 1:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type in ["alone"]:
		locs = []
		all_locs = []
		other_locs = []

		for subj in subjects.values():
			locs.append((subj["x"], subj["y"]))

		for ent_id, ent in entities.items():
			all_locs.append((ent["x"], ent["y"]))
			if ent_id not in subjects.keys():
				other_locs.append((ent["x"], ent["y"]))

		if 0 < len(locs) and len(locs) < 7:
			valid = True
		else:
			return False, False, None

		subj_shortest_dist = []
		for loc in locs:
			shortest_dist = 10000
			for other_loc in other_locs:
				shortest_dist = min(shortest_dist, compute_distance(loc, other_loc))
			subj_shortest_dist.append(shortest_dist)

		all_shortest_dist = []
		for loc_1 in all_locs:
			shortest_dist = 10000
			for loc_2 in all_locs:
				if loc_1 != loc_2:
					shortest_dist = min(shortest_dist, compute_distance(loc_1, loc_2))
			all_shortest_dist.append(shortest_dist)

		value = np.mean(subj_shortest_dist) / np.mean(all_shortest_dist)

		if value > 1:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "far":
		subj_locs = []
		obj_locs = []
		all_locs = []
		other_locs = []

		for subj in subjects.values():
			subj_locs.append((subj["x"], subj["y"]))

		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_locs.append((obj["x"], obj["y"]))

		for ent_id, ent in entities.items():
			all_locs.append((ent["x"], ent["y"]))
			if ent_id not in subjects.keys():
				other_locs.append((ent["x"], ent["y"]))

		if 0 < len(subj_locs) and len(subj_locs) < 7:
			valid = True
		else:
			return False, False, None

		if len(obj_locs) > 0:
			subj_shortest_dist = []
			for subj_loc in subj_locs:
				shortest_dist = 10000
				for obj_loc in obj_locs:
					shortest_dist = min(shortest_dist, compute_distance(subj_loc, obj_loc))
				subj_shortest_dist.append(shortest_dist)
		else:
			subj_shortest_dist = []
			for subj_loc in subj_locs:
				shortest_dist = 10000
				for other_loc in other_locs:
					shortest_dist = min(shortest_dist, compute_distance(subj_loc, other_loc))
				subj_shortest_dist.append(shortest_dist)

		all_shortest_dist = []
		for loc_1 in all_locs:
			shortest_dist = 10000
			for loc_2 in all_locs:
				if loc_1 != loc_2:
					shortest_dist = min(shortest_dist, compute_distance(loc_1, loc_2))
			all_shortest_dist.append(shortest_dist)

		value = np.mean(subj_shortest_dist) / np.mean(all_shortest_dist)

		if value > 1:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	"""
		Check relations for region.
	"""

	if relation_type == "interior":
		subj_xs = []
		subj_ys = []
		for subj in subjects.values():
			subj_xs.append(subj["x"])
			subj_ys.append(subj["y"])

		obj_xs = []
		obj_ys = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_xs.append(obj["x"])
				obj_ys.append(obj["y"])

		if no_object:
			if len(subj_xs) > 0:
				valid = True
			else:
				return False, False, None

			satisfy = True
			for subj_x, subj_y in zip(subj_xs, subj_ys):
				if compute_distance((subj_x, subj_y), (base_x, base_y)) >= 120:
					satisfy = False
		else:
			if len(subj_xs) > 0 and len(obj_xs) > 1:
				valid = True
			else:
				return False, False, None

			satisfy = True
			for subj_x, subj_y in zip(subj_xs, subj_ys):
				if (subj_x < min(obj_xs) or max(obj_xs) < subj_x) \
					and (subj_y < min(obj_ys) or max(obj_ys) < subj_y):
					satisfy = False

		return valid, satisfy, None

	if relation_type == "exterior":
		subj_xs = []
		subj_ys = []
		for subj in subjects.values():
			subj_xs.append(subj["x"])
			subj_ys.append(subj["y"])

		obj_xs = []
		obj_ys = []
		for obj in objects.values():
			obj_xs.append(obj["x"])
			obj_ys.append(obj["y"])

		if no_object:
			if len(subj_xs) > 0:
				valid = True
			else:
				return False, False, None

			satisfy = True
			for subj_x, subj_y in zip(subj_xs, subj_ys):
				if compute_distance((subj_x, subj_y), (base_x, base_y)) < 120:
					satisfy = False
		else:
			if len(subj_xs) > 0 and len(obj_xs) > 0:
				valid = True
			else:
				return False, False, None

			satisfy = True
			for subj_x, subj_y in zip(subj_xs, subj_ys):
				if (min(obj_xs) < subj_x and subj_x < max(obj_xs)) \
					and (min(obj_ys) < subj_y and subj_y < max(obj_ys)):
					satisfy = False

		return valid, satisfy, None

	"""
		Check relations for color.
	"""
	if relation_type == "darker":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_colors = []
		for subj in subjects.values():
			subj_colors.append(extract_color(subj["color"]))

		obj_colors = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_colors.append(extract_color(obj["color"]))

		all_colors = []
		for ent in entities.values():
			all_colors.append(extract_color(ent["color"]))

		if len(obj_colors) > 0:
			value = np.mean(subj_colors) - np.mean(obj_colors)
		else:
			value = np.mean(subj_colors) - np.mean(all_colors)

		if value < 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "darkest":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_colors = []
		for subj in subjects.values():
			subj_colors.append(extract_color(subj["color"]))

		obj_colors = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_colors.append(extract_color(obj["color"]))

		other_colors = []
		for ent_id, ent in entities.items():
			if ent_id not in subjects.keys():
				other_colors.append(extract_color(ent["color"]))

		value = 0
		if len(obj_colors) > 0:
			for obj_color in obj_colors:
				if obj_color < np.mean(subj_colors):
					value = max(value, np.mean(subj_colors) - obj_color)
		else:
			for other_color in other_colors:
				if other_color < np.mean(subj_colors):
					value = max(value, np.mean(subj_colors) - other_color)

		if value == 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "lighter":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_colors = []
		for subj in subjects.values():
			subj_colors.append(extract_color(subj["color"]))

		obj_colors = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_colors.append(extract_color(obj["color"]))

		all_colors = []
		for ent in entities.values():
			all_colors.append(extract_color(ent["color"]))

		if len(obj_colors) > 0:
			value = np.mean(subj_colors) - np.mean(obj_colors)
		else:
			value = np.mean(subj_colors) - np.mean(all_colors)

		if value > 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "lightest":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_colors = []
		for subj in subjects.values():
			subj_colors.append(extract_color(subj["color"]))

		obj_colors = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_colors.append(extract_color(obj["color"]))

		other_colors = []
		for ent_id, ent in entities.items():
			if ent_id not in subjects.keys():
				other_colors.append(extract_color(ent["color"]))

		value = 0
		if len(obj_colors) > 0:
			for obj_color in obj_colors:
				if obj_color > np.mean(subj_colors):
					value = max(value, obj_color - np.mean(subj_colors))
		else:
			for other_color in other_colors:
				if other_color > np.mean(subj_colors):
					value = max(value, other_color - np.mean(subj_colors))

		if value == 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "same color":
		colors = []

		for subj in subjects.values():
			colors.append(extract_color(subj["color"]))

		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				colors.append(extract_color(obj["color"]))

		if len(colors) > 1:
			valid = True
		else:
			return False, False, None

		value = np.max(colors) - np.min(colors)

		if value < 30:
			satisfy = True	
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "different color":
		colors = []

		for subj in subjects.values():
			colors.append(extract_color(subj["color"]))

		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				colors.append(extract_color(obj["color"]))

		if len(colors) > 1:
			valid = True
		else:
			return False, False, None

		value = np.max(colors) - np.min(colors)

		if value > 30:
			satisfy = True	
		else:
			satisfy = False

		return valid, satisfy, value

	"""
		Check relations for size.
	"""
	if relation_type == "smaller":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_sizes = []
		for subj in subjects.values():
			subj_sizes.append(subj["size"])

		obj_sizes = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_sizes.append(obj["size"])

		all_sizes = []
		for ent in entities.values():
			all_sizes.append(ent["size"])

		value = 0
		if len(obj_sizes) > 0:
			value = np.mean(subj_sizes) - np.mean(obj_sizes)
		else:
			value = np.mean(subj_sizes) - np.mean(all_sizes)

		if value < 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "smallest":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_sizes = []
		for subj in subjects.values():
			subj_sizes.append(subj["size"])

		obj_sizes = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_sizes.append(obj["size"])

		other_sizes = []
		for ent_id, ent in entities.items():
			if not ent_id in subjects:
				other_sizes.append(ent["size"])

		value = 0
		if len(obj_sizes) > 0:
			for obj_size in obj_sizes:
				if np.mean(subj_sizes) > obj_size:
					value = max(value, np.mean(subj_sizes) - obj_size)
		else:
			for other_size in other_sizes:
				if np.mean(subj_sizes) > other_size:
					value = max(value, np.mean(subj_sizes) - other_size)

		if value == 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "larger":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_sizes = []
		for subj in subjects.values():
			subj_sizes.append(subj["size"])

		obj_sizes = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_sizes.append(obj["size"])

		all_sizes = []
		for ent in entities.values():
			all_sizes.append(ent["size"])

		value = 0
		if len(obj_sizes) > 0:
			value = np.mean(subj_sizes) - np.mean(obj_sizes)
		else:
			value = np.mean(subj_sizes) - np.mean(all_sizes)

		if value > 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "largest":
		if len(subjects) > 0:
			valid = True
		else:
			return False, False, None

		subj_sizes = []
		for subj in subjects.values():
			subj_sizes.append(subj["size"])

		obj_sizes = []
		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				obj_sizes.append(obj["size"])

		other_sizes = []
		for ent_id, ent in entities.items():
			if not ent_id in subjects:
				other_sizes.append(ent["size"])

		value = 0
		if len(obj_sizes) > 0:
			for obj_size in obj_sizes:
				if np.mean(subj_sizes) < obj_size:
					value = max(value, obj_size - np.mean(subj_sizes))
		else:
			for other_size in other_sizes:
				if np.mean(subj_sizes) < other_size:
					value = max(value, other_size - np.mean(subj_sizes))

		if value == 0:
			satisfy = True
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "same size":
		sizes = []

		for subj in subjects.values():
			sizes.append(subj["size"])

		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				sizes.append(obj["size"])

		if len(sizes) > 1:
			valid = True
		else:
			return False, False, None

		value = np.max(sizes) - np.min(sizes)

		if value <= 2:
			satisfy = True	
		else:
			satisfy = False

		return valid, satisfy, value

	if relation_type == "different size":
		sizes = []

		for subj in subjects.values():
			sizes.append(subj["size"])

		for obj_id, obj in objects.items():
			if obj_id not in subjects.keys():
				sizes.append(obj["size"])

		if len(sizes) > 2:
			valid = True
		else:
			return False, False, None

		value = np.max(sizes) - np.min(sizes)

		if value > 1:
			satisfy = True	
		else:
			satisfy = False

		return valid, satisfy, value
