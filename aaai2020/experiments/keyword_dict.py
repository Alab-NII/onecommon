keyword_dict = {}
replace_word_dict = {}
misspelling_dict = {}

keyword_dict["special"] = ['<eos>', '<unk>', '<selection>', '<pad>']

# words that occur >= 10 times in the corpus
keyword_dict["common"] = ['the', 'THEM:', 'YOU:', 'a', 'i', 'dot', 'one', 'gray', ',', 'is', 'have', 'to', 'and', '.', 'black', 'of', 'large', 'do', 'dark', 'that', 'dots', 'see', 'right', 'light', 'yes', 'it', 'medium', 'you', 'two', 'left', 'small', 'on', 'click', 'with', 'smaller', 'in', "'s", 'top', 'slightly', 'very', 'are', 'larger', 'darker', 'three', 'ok', 'bottom', 'above', 'at', 'lighter', 'same', 'tiny', 'let', 'size', 'close', 'other', 'below', 'no', "n't", 'choose', 'there', '?', 'but', 'my', 'lets', 'about', 'almost', '!', 'middle', 'circle', 'line', 'select', 'its', 'not', 'lower', 'think', 'we', '...', 'triangle', 'all', 'near', 'than', 'sized', 'each', 'they', 'up', 'by', 'together', 'your', 'largest', 'darkest', 'down', "o'clock", 'four', 'from', 'so', 'little', 'center', 'smallest', 'bigger', 'color', 'side', 'next', 'ones', 'what', 'if', 'just', 'upper', 'only', 'those', 'mine', 'too', 'also', 'lightest', 'which', 'cluster', 'for', 'different', 'be', 'like', 'far', 'them', 'shade', 'an', 'another', 'bit', 'then', 'or', 'want', 'really', 'as', 'got', 'edge', 'any', 'towards', 'vertical', 'off', 'lonely', 'should', 'has', 'me', 'how', 'itself', 'diagonal', 'try', 'go', 'both', 'does', 'directly', 'similar', '(', 'five', 'hmm', 'between', 'us', 'kind', 'being', 'sounds', 'u', 'group', 'away', 'hi', 'where', 'can', 'form', 'pretty', 'rest', ')', 'much', 'higher', 'going', 'horizontal', 'alone', 'inch', 'forming', 'hello', 'pair', 'touching', 'around', 'circles', 'this', 'good', 'our', 'well', 'seven', 'toward', 'under', 'sure', 'six', 'more', 'colored', 'take', "'re", 'would', 'biggest', 'make', 'others', "'m", 'way', 'talking', 'shape', 'yours', 'will', 'out', 'most', 'did', 'furthest', 'first', 'lone', 'aligned', '9', 'nothing', 'maybe', 'apart', 'shall', 'great', 'anything', 'might', 'position', ':', 'further', 'identical', 'closer', 'making', 'was', 'kinda', 'screen', 'sorry', '10', '11', 'half', 'slight', 'exactly', 'lowest', 'either', 'view', 'described', 'may', 'closest', 'could', 'row', 'look', '12', 'sizes', 'vertically', '8', 'dotted', 'across', 'many', "'ve", 'shades', 'located', 'am', 'oh', 'nearly', 'beside', 'fairly', 'second', 'views', 'looks', 'straight', 'sort', 'mentioned', 'thing', 'cut', 'single', 'part', 'spot', 'corner', 'border', 'perfect', 'horizontally', 'say', 'diagonally', "'ll", 'own', 'haha', 'set', 'quite', 'believe', 'even', 'describe', 'mean', 'common', 'because', 'point', 'correct', 'underneath', 'though', 'sloping', 'south', 'know', 'l', 'north', 'somewhat', 'seeing', 'looking', 'said', 'wait', 'here', 'inches', 'some', 'else', 'clustered', 'angle', 'actually', 'their', 'were', 'third', 'over', 'alright', 'pale', 'meant', 'pm', 'super', 'smallish', 'extremely', 'isolated', 'east', 'chose', 'hand', 'centered', 'colors', 'tiniest', 'c', 'except', 'description', 'pairs', 'bunch', 'time', 'last', 'distance', 'hit', 'lot', 'highest', "''", '``', 'beneath', 'guess', 'something', 'shaded', 'equal', 'spread', 'nose', 'makes', 'less', 'these', 'upside', 'exact', 'ish', 'now', 'blackest', 'few', 'semi', 'ah', "'", 'clock', 'darkish', 'probably', 'grouping', 'slanted', 'sets', 'clicking', 'selected', 'still', ';', 'get', 'opposite', 'cool', 'perfectly', 'square', 'none', 'match', 'end', 'massive', 'face', 'tell', 'selecting', 'referring', 'seems', 'tight', 'mark', 'northeast', 'diamond', 'southwest', 'quadrant', 'mostly', 'forms', 'southeast', 'west', 'luck', 'outer', 'spaced', '3:00', 'relation', 'greys', 'please', 'y', 'barely', 'original', 'pointing', 'work', 'o', 'starting', 'surrounded', 'nice', '2:00', 'v', 'describing', 'letter', 'before', 'choosing', 'ca', 'back', 'hey', 'real', 'northwest', 'nearby', 'positioned', 'must', 'outside', 'seem', 'ways', 'eye', 'wow', 'outline', 'read', '=', 'low', 'stacked', 'talked', 'equally', 'lines', 'lined', 'themselves', 'least', 'confirm', 'immediately', 'everything', '6:00', 'message', 'immediate', 'wo', 'area', 'thanks', 'type', 'missing', 'location', 'couple', 'parallel', 'pattern', 'sound', 'several', 'trapezoid', 'triangles', 'made', 'upwards', 'goes', 'compared', 'groups', 'hair', 'awesome', 'sorta', 'halfway', 'along', 'pure', 'space', 'long', 'high', 'touch', "'d", 'anywhere', 'need', '7:00', 'spots', 'eyes', 'mines', 'approximately', '5:00', 'fourth', 'sides', 'typing', 'roughly', 'relative', 'total', 'had', 'earlier', '9:00', 'sitting', 'scattered', 'positions', 'thank', 'picking', 'blackish', 'dead', 'basically', 'rectangle', 'lightish', 'curve', 'topmost', 'inside', 'formation', 'grouped', 'shaped', 'backwards', 'clicked', 'edges', '12:00', 'places', 'find', 'angled', 'closely', 'differently', 'running', 'adjacent', 'possibly', 'picked', 'level', 'dipper', 'darkness', 'fat', 'boomerang', 'place', '10:00', 'normal', 'cm', 'curved', 'relatively', 'found', '1:00', '11:00', 'picture', '8:00', 'upward', 'definitely', 'always', 'give', 'help', 'past', 'check', 'after', 'minute', 'completely', 'cuts', 'angling', 'wanna', 'agree', 'slighter', 'while', 'lots', 'locations', 'when', 't', 'saying', 'since', 'slope', 'regular', 'hard', 'offset', 'friend', 'various', 'points', 'use', 'saw', 'bet', 'hope', 'question', 'twelve', 'slant', 'tip', 'varying', 'centimeter', 'obviously', 'speak', 'direction', 'equidistant', 'call', 'grays', 'm', 'ends', 'rather', '4:00', 'among', 'easy', 'start', 'kite']
keyword_dict["all"] = keyword_dict["special"] + keyword_dict["common"]

keyword_dict["normal_color"] = ["gray", "dark", "black", "dark", "light", "pale", "darkish"]

keyword_dict["comparative_color"] = ["darker", "lighter"]
replace_word_dict["comparative_color"] = ["lower"]

keyword_dict["superlative_color"] = ["darkest", "lightest", "blackest"]
replace_word_dict["superlative_color"] = ["lowest"]

keyword_dict["color"] = keyword_dict["normal_color"] + keyword_dict["comparative_color"] + keyword_dict["superlative_color"]

keyword_dict["normal_size"] = ["large", "small", "big", "tiny", "huge", "massive"]
replace_word_dict["size"] = ["bottom"]

keyword_dict["comparative_size"] = ["smaller", "larger", "bigger"]
replace_word_dict["comparative_size"] = ["lower"]

keyword_dict["superlative_size"] = ["largest", "smallest", "biggest", "tiniest"]
replace_word_dict["superlative_size"] = ["lowest"]

keyword_dict["size"] = keyword_dict["normal_size"] + keyword_dict["comparative_size"] + keyword_dict["superlative_size"]

keyword_dict["normal_location"] = ["right", "left", "top", "bottom", "middle", "up", "center", "down", "north", "east", "south", "west", "northwest", "northeast", "southwest", "southeast"]
replace_word_dict["location"] = ["bottom", "down", "south"]

keyword_dict["comparative_location"] = ["lower", "upper", "higher"]
replace_word_dict["comparative_location"] = ["lower"]

keyword_dict["superlative_location"] = ["lowest", "topmost"]
replace_word_dict["superlative_location"] = ["lowest"]

keyword_dict["location"] = keyword_dict["normal_location"] + keyword_dict["comparative_location"] + keyword_dict["superlative_location"]

keyword_dict["area"] = ["edge", "side"]

keyword_dict["direction"] = ["vertical", "diagonal", "horizontal"]

keyword_dict["direction_adv"] = ["vertically", "horizontally", "diagonally"]

keyword_dict["grouping"] = ["line", "triangle", "cluster", "square", "diamond", "trapezoid", "group", "kite", "pair"]

#keyword_dict["spatial_relation"] = ["above", "below", "close", "line", "triangle", "near", "together", "cluster", "far", "vertical", "diagonal", "horizontal", "under", "closer", "furthest", "vertically", "closest", "horizontally", "diagonally", "clustered", "underneath", "over", ]

keyword_dict["proximity"] = ["close", "together", "clustered", "grouped", "lonely", "alone", "touching", "apart", "lone", "isolated", "loner"]

keyword_dict["comparative_proximity"] = ["further", "closer"]

keyword_dict["superlative_proximity"] = ["furthest", "closest", "farther"]

keyword_dict["negation"] = ["dont", "n't", "nothing", "doesnt"]

keyword_dict["conjunction"] = ["and", "but", "or"]

keyword_dict["quantifier"] = ["any", "all", "each", "both", "none", "most", "many", "some", "few", "several"]

#keyword_dict["preposition"] = ["on", "with", "in", "above", "at", "below", "near", "by", "from", "for", "as", "towards", "off", "away", "between", "around", "toward", "under", "out", "across", "beside", "underneath", "over", "before", "inside", "after"]
keyword_dict["preposition"] = ["on", "with", "in", "above", "at", "below", "near", "by", "from", "for", "towards", "off", "away", "between", "around", "toward", "under", "out", "across", "beside", "underneath", "over", "inside"]
replace_word_dict["preposition"] = ["below", "under"]

keyword_dict["approximation"] = ["almost", "nearly", "roughly", "med"]

keyword_dict["exactness"] = ["directly", "exact", "exactly", "immediate", "immediately", "perfect", "perfectly", "pure"]

keyword_dict["approximation/exactness"] = keyword_dict["approximation"] + keyword_dict["exactness"]
replace_word_dict["approximation/exactness"] = ["exactly"]

keyword_dict["subtlety"] = ["barely", "bit", "slight", "slightly", "somewhat"]

keyword_dict["extremity"] = ["very", "really", "super", "extremely", "quite", "dead", "much", "pretty"]

keyword_dict["degree"] = keyword_dict["subtlety"] + keyword_dict["extremity"]
replace_word_dict["degree"] = ["very", "really", "super", "extremely"]

keyword_dict["uncertainty"] = ["believe", "guess", "kinda", "may", "maybe", "might", "possibly", "probably", "sort", "sorta", "think", "could"]

keyword_dict["number_word"] = ["one", "two", "three", "four", "five", "seven"]

keyword_dict["number_rank"] = ["first", "second", "third", "last", "2nd"]

keyword_dict["number"] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "11", "10", "12"]

keyword_dict["yes_no"] = ["yeah", "yes", "yep", "yup", "no", "nope", "na"]
replace_word_dict["yes_no"] = ["no"]

keyword_dict["acknowledgement"] = ["ok", "okay", "k", "alright", "cool", "yea"]

keyword_dict["similarity"] = ["same", "different", "similar", "identical", "opposite"]

keyword_dict["similarity_adv"] = ["differently", "equally"]

keyword_dict["me_you"] = ["i", "me", "you", "u"]

keyword_dict["my_your"] = ["my", "your", "ur"]

keyword_dict["mine_yours"] = ["mine", "yours", "mines"]

keyword_dict["interjection"] = ["hi", "hello", "hm", "hmm", "hmmm", "...", "sorry", "oh", "lol", "ah", "hey", "haha", "wow", "thank", ]

keyword_dict["comparative"] = ["more"]

keyword_dict["superlative"] = ["least", "most"]

keyword_dict["polysemy"] = ["about", "little", "near", "just", "right", "like", "no", "rest", "well", "kind"]




keyword_dict["relation"] = keyword_dict["comparative_color"] + keyword_dict["superlative_color"] + keyword_dict["comparative_size"] + keyword_dict["superlative_size"] + keyword_dict["direction"] + keyword_dict["direction_adv"] + keyword_dict["grouping"] + keyword_dict["proximity"] + keyword_dict["comparative_proximity"] + keyword_dict["superlative_proximity"] + keyword_dict["similarity"] + keyword_dict["similarity_adv"]

#keyword_dict["content"] = keyword_dict["color"] + keyword_dict["size"] + keyword_dict["direction"] + keyword_dict["direction_adv"] + keyword_dict["grouping"] + keyword_dict["proximity"] + keyword_dict["comparative_proximity"] + keyword_dict["superlative_proximity"] + keyword_dict["similarity"] + keyword_dict["similarity_adv"]

keyword_dict["content"] = ['one', 'gray', 'black', 'large', 'dark', 'right', 'light', 'yes', 'medium', 'two', 'left', 'small', 'smaller', 'top', 'slightly', 'larger', 'darker', 'three', 'bottom', 'above', 'lighter', 'same', 'tiny', 'close', 'below', 'no', "n't", 'middle', 'line', 'not', 'lower', 'triangle', 'all', 'together', 'largest', 'darkest', 'down', 'four', 'little', 'center', 'smallest', 'bigger', 'upper', 'lightest', 'cluster', 'different', 'far', 'vertical', 'lonely', 'diagonal', 'directly', 'similar', 'five', 'between', 'group', 'higher', 'horizontal', 'alone', 'pair', 'touching', 'seven', 'six', 'biggest', 'furthest', 'first', 'lone', 'aligned', '9', 'nothing', 'apart', 'further', 'identical', 'closer', '10', '11', 'lowest', 'closest', '12', 'vertically', '8', 'straight', 'corner', 'border', 'horizontally', 'diagonally', "'ll", 'sloping', 'south', 'north', 'clustered', 'angle', 'third', 'pale', 'smallish', 'isolated', 'east', 'centered', 'tiniest', 'c', 'pairs', 'bunch', 'highest', 'equal', 'spread', 'nose', 'blackest', 'darkish', 'grouping', 'slanted', 'opposite', 'square', 'none', 'end', 'massive', 'face', 'northeast', 'diamond', 'southwest', 'quadrant', 'southeast', 'west', '3:00', 'greys', 'y', 'o', '2:00', 'v', 'northwest', 'eye', 'low', 'stacked', 'equally', 'lines', 'lined', 'least', '6:00', 'couple', 'parallel', 'trapezoid', 'triangles']
