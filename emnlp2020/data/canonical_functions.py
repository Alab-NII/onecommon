import numpy as np
from sklearn.linear_model import LinearRegression
import pdb
from nltk import word_tokenize, pos_tag, bigrams, ngrams
from itertools import permutations, combinations 


# Dictionary of canonical functions
canonical_functions = {}

canonical_functions["Mod-Subtlety"] = {"slight", "slightly", "little", "bit", "barely", "faintly", "subtle", "subtly"}
canonical_functions["Mod-Extremity"] = {"very", "much", "extremely", "quite", "super", "pretty"}
canonical_functions["Mod-Certainty-Exactness"] = {"exactly", "directly", "absoultely", "clear", "clearly", "complete", "completely", "definite", "definitly", "direct", "directly", "exact", "exactly", "perfect", "perfectly", "totally"}
canonical_functions["Mod-Uncertainty-Approximation"] = {"almost", "about", "kinda", "sorta", "maybe", "might", "perhaps", "possibly", "probably", ("kind", "of"), ("sort", "of")}
canonical_functions["Mod-Neutral"] = {"medium", "med", "moderately", "fairly"}
canonical_functions["Mod-Negation"] = {"not"}

def detect_canonical_function(modifier_text):
	detected_canonical_function = None

	tokens = word_tokenize(modifier_text.lower())

	# automatically detect canonical function
	for canonical_function in canonical_functions.keys():
		key_words = canonical_functions[canonical_function]
		for key_word in key_words:
			if isinstance(key_word, tuple):
				if key_word in ngrams(tokens, len(key_word)):
					detected_canonical_function = canonical_function
					break
			else:
				if key_word in tokens:
					detected_canonical_function = canonical_function
					break

	if detected_canonical_function:
		return detected_canonical_function
	else:
		#return unknown if nothing was detected
		return "Mod-Undefined"
