# Section 3. Canonicalization

In this section, we will explain how to annotate the canonical relations of spatial relations, as well as canonical functions of modifiers.

## Canonical Relations

We consider the following 5 categories of 24 canonical relations in our annotation:

**Direction/Orientation**

1. left (Ex. *left*, *west*)

2. right (Ex. *right*, *east*)

3. above (Ex. *above*, *up*, *top*, *high*, *higher*, *north*)

4. below (Ex. *below*, *down*, *beneath*, *bottom*, *low*, *lower*, *under*, *south*)

5. horizontal (Ex. *horizontal*, *horizontally*)

6. vertical (Ex. *vertical*, *vertically*)

7. diagonal (Ex. *diagonal*, *diagonally*, *slanted*, *slope*, *sloping*)

**Distance/Proximity**

8. near (Ex. *near*, *close*, *cluster*, *clustered*, *together*)

9. far (Ex. *far*, *away*, *apart from*, *separated*, *further*, *furthest*)

10. alone (Ex. *alone*, *lonesome*, *lone*, *lonely*, *isolated*, *solo*)

**Region**

11. interior (Ex. *middle*, *center*, *between*)

12. exterior (Ex. *outside*, *edge*, *border*)

**Color Comparison**

13. lighter (Ex. *lighter*)

14. lightest (Ex. *lightest*)

15. darker (Ex. *darker*)

16. darkest (Ex. *darkest*)

17. same color (Ex. *same color*, *same colored*, *similar color*, *similar colored*)

18. different color (Ex. *different color*, *different colored*, *different shade*, *different shaded*, *opposite color*, *opposite colored*)

**Size Comparison**

19. smaller (Ex. *smaller*, *tinier*)

20. smallest (Ex. *smallest*, *tiniest*)

21. larger (Ex. *larger*, *bigger*)

22. largest (Ex. *largest*, *biggest*)

23. same size (Ex. *same size*, *same sized*, *similar size*, *similar sized*)

24. different size (Ex. *different size*, *different sized*, *opposite size*, *opposite sized*)

Note that multiple canonical relations can be implied by a single expression

- *"on the bottom left"* (below, left)

- *"identical dots"* (same color, same size)

- *"different shades and sizes"* (different color, different size)

or none of them can be implied, e.g.

- *"is part of"*, *"among"* (topological relations)

- *"form a line"*, *"triangle"*, *"in a very flat shape of V"* (shapes)

- *"second darkest"*

- *"equal distance from"*

- *"missing in my circle"*

We only annotate canonical relations that are **clearly implied**. For ambiguous cases, we propose the following policy (which can be ignored depending on context):

Consider as **implied**:

- *"toward the top"*, *"near the top"* (above)

- *"on top of"* each other, *"above"* each other (vertical)

- *"just"* to the left, *"right"* below, *"immediate"* left, *"beside"* (near)

- *"trio"*, *"pair"* (near)

- *"side by side"* (near, horizontal)

- (>=2) inches away (far)

- (<=0.5) inch away (near)

- *"by"* itself, *"off by"* itself, *"far off by"* itself (alone)

- *"close to the center"*, *"near center"* (interior)

- *"close to border"*, *"touching the dotted line"* (exterior)

- *"surrounded by"* (interior)

- all *"around"* it, *"on each side of"*, *"surrounding"* (exterior)

- *"large compared to"*, *"double the size of"* (larger)

Consider as **not implied** (without additional information):

- *"diamond"*, *"triangle"*, *"square"* (?near)

- *"pairs"*, *"groups"*, *"lines"* (?near)

- *"closer to"*, *"closest to"* (?near)

- *"closer to the center"*, *"closest to the center"* (?center)

- *"further away"*, *"furthest from"*, *"furthest left"*, *"far right in the cluster"* (?far)

- *"far apart"*, *"spread out"* (?far)

- *"with"*, *"along with"* *"have"*, *"has"* (?near)

- *"next to"*, *"beside"* (?horizontal)

- *"middle left"*, *"left of center"* *"to the left center"* (?middle)

- *"your left"*, *"your tiniest"* (?left, ?tiniest)

- *"in"*, *"within"* (?interior)

- *"off center"* (?exterior)

- *"diagonal L Shape"*, *"triangle leaning to the left"* (?diagonal)

- two lines same shape *"down and to the left"* (?diagonal)

- *"curving from left to right downward"* (?diagonal)

- *"across the middle of the circle"* (?middle)

## Canonical Functions

As discussed in [Section 1. span detection](../1_span_detection), we consider the following 6 types of canonical functions:

1. Subtlty

- *"slightly"*, *"a bit"*, *"a little"*

2. Uncertainty/Approximation

- *"maybe"*, *"could be"*, *"sort of"*, *"kind of"*, *"almost"*, *"mostly"*

3. Extremity

- *"very"*, *"extremely"*, *"super"*

4. Certainty/Exactness

- *"perfectly"*, *"completely"*, *"definitely"*

5. Neutrality

- *"medium"*, *"moderately"*

6. Negation

- *"not"*

For compound/holistic modifiers, the overall function that is implied by the modification should be annotated, e.g.

- *very slightly* (Subtlty)

- *not completely* (Ambiguity/Uncertainty)

- *almost perfectly* (Certainty/Exactness)

- *seems pretty* (Extremity)

Finally, we focus on canonical functions that are **clearly** implied. For ambiguous cases, we propose the following policy (which can be ignored depending on context):

Consider as implied:

- *"tight"* cluster (Extremity)

- *"loose"* cluster (Subtlety)

- *"quite"* (Extremity)

- *"all the way"* at the top (Extremity)

- *"could"* make a triangle (Uncertainty/Approximation)

- *"right"* across the equator, *"right"* in the middle (Certainty/Exactness)

- *"all"* by itself (Certainty/Exactness)

- *"about"* 3 o'clock (Uncertainty/Approximation)

- *"almost"* form a *"perfect"* triangle (Uncertainty/Approximation, Certainty/Exactness)

- *"medium"* to *"a bit"* large (Neutral, Subtlety)

- *"a shade"* lighter (Subtlety)

Consider as **not** implied (without additional information):

- *"near"* middle of the circle, *"around"* 3 o'clock (?Uncertainty/Approximation)

- *"all"* around it (?Certainty/Exactness)

- *"like"* that (?Uncertainty/Approximation)

## Annotation Tips

There is no need to annotate complex directions (*diagonal*, *vertical*, *horizontal*) when simple directions (*left*, *right*, *above*, *below*) are annotated.

- *"11 o'clock"* (left, above)

- *"12 o'clock"* (above)

- *straight above* (above)

To reduce the annotation effort, we automatically detect canonical relations if the spatial relation includes the following keywords (unigrams and bigrams).

In cases where the automatic detection fails (it detects wrong relations or does not detect correct relations), the annotator should use brat's **annotator note** to write down simple paraphrases which can be used instead for automatic detection.

Automatic detection success:

- *"lighter than"* (*lighter* is correctly detected)

Automatic detection fails:

- sloping up to the right (wrongly detects *above* and *right*)

- southwest (does not detect *below* and *left*)

If general, annotator should manually annotate the correct canonical relations (instead of relying on automatic detection).

```
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
```

Similarly, we automatically detect canonical functions if the modifier includes the following keywords (unigrams and bigrams).

```
# Dictionary of canonical functions
canonical_functions = {}

canonical_functions["Mod-Subtlety"] = {"slight", "slightly", "little", "bit", "barely", "faintly", "subtle", "subtly"}
canonical_functions["Mod-Extremity"] = {"very", "much", "extremely", "quite", "super", "pretty"}
canonical_functions["Mod-Certainty-Exactness"] = {"exactly", "directly", "absoultely", "clear", "clearly", "complete", "completely", "definite", "definitly", "direct", "directly", "exact", "exactly", "perfect", "perfectly", "totally"}
canonical_functions["Mod-Uncertainty-Approximation"] = {"almost", "about", "kinda", "sorta", "maybe", "might", "perhaps", "possibly", "probably", ("kind", "of"), ("sort", "of")}
canonical_functions["Mod-Neutral"] = {"medium", "med", "moderately", "fairly"}
canonical_functions["Mod-Negation"] = {"not"}
```