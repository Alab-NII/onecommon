# Section 1. Span Detection

In this section, we will explain how to annotate the span of spatial attributes, spatial relations and modifiers.

## Spatial Attributes

We consider entity-level information without explicit comparison as **spatial attributes**. These typically include expressions of color (*"dark gray"*, *"black"*), size (*"large"*, *"small"*) and numbers (*"two"*, *"three"*).

Spatial attributes are typically expressed as

Adjectives:

- My dots are *"dark"*.

Adjectival phrases:

- Is there a dot *"of dark color"*?

Nominals:

- Left one is *"a black dot"*.

To reduce the annotation effort, we do not annotate spatial attributes *inside* markables (since they can be easily detected with obvious argument structures).

Expressions like copulas (*is*, *being*), articles (*a*, *the*), particles (*to*, *with*) and modifiers can be omitted or included.

- Left one *"being black"* = Left one being *"black"*.

- Dot with *"black color"* = Dot *"with black color"*

In cases of implicit/explicit conjunctions, spatial attributes can be annotated jointly if and only if they have the same arguments and modifiers.

Jointly annotatable:

- My dot is *"dark and large"* = My dot is *"dark"* and *large"*

- Left being *"large dark"* = Left being *"large"* *"dark"*

- ... which contains *"five dark"* dots = ... which contains *"five"* *"dark"* dots

Jointly unannotatable:

- Top dot is very *"dark"* but slightly *"small"* (different modifiers)

Holistic/compound spatial attributes should be annotated as a single expression.

Must be annotated jointly:

- They are *"medium in size and color"*.

- I would say *"medium to small"*

Rare/complex spatial attributes should also be annotated:

- I have two dots *"of that color"*. (anaphoric expressions)

- They are *"all shades of gray"*. (complex expressions)

Finally, keep in mind the predicate-argument structure of spatial attributes, and do the best to annotate spans that have *existing markables* as the arguments.

- The dots are *"light and dark gray"*. (*"light"* and *"dark gray"* should not be annotated independently since they have no corresponding subjects)

### Spatial Relations

We consider locational expressions and explicit comparison of spatial attributes as **spatial relations**. These typically include expressions of direction (*"left of"*, *"above"*), proximity (*"near"*, *"clustered"*), region (*"between"*, *"outside"*), topological relations (*"in"*, *"contains"*) and color/size/number comparisons (*"darker"*, *"larger"*, *"more"*).

Spatial relations are typically expressed as:

Prepositions and particles

- *on*, *in*, *to the right of*, *by*, *among*, *beneath*, *with*

Verbs with location-related information

- *clustered*, *isolated*, *angling up*

Nouns and Noun phrases

- *triangle*, *group*, *line*

Adjectives and adjectival phrases

- *alone*, *darker*, *darkest*, *identical*

- *of the same size*, *at 7 o'clock*

Since their arguments are usually non-obvious, we annotate all spatial relations inside and outside markables.

Copulas, articles and modifiers can be either omitted or included. In cases of implicit/explicit conjunctions, spatial relations can be annotated jointly if they have the same arguments and modifiers.

Jointly annotatable:

- It is *"on the left and above"* my dark dot = It is *"on the left"* and *"above"* my dark dot

- Black is *"bottom left"* = Black is *"bottom"* *"left"*

Jointly unannotatable:

- Top dot is *"darker"* and *"slightly smaller"* (different modifiers)

Must be annotated jointly:

- Three dots are *"sloping up and to the right"* (holistic/compound expression)

Rare/complex spatial relations should also be annotated, e.g. 

- One pair is much *"closer than"* the other pair.

- Two large dots *"on each side of"* the dot.

- Yes, they are *"in those placements"*.

You do not need to annotate implicit relations (even if it's inferrable):

- Small dot to the right of a large dot (don't annotate *"smaller"* or *"larger"*)

Finally, keep in mind the predicate-argument structure of spatial relations, and do the best to annotate spans that have *existing makrables* as arguments.

### Modifiers

We consider **modifiers** as expressions that modify spatial expressions with the following functions:

1. Subtlty

- *"slightly"*, *"a bit"*, *"a little"*

2. Ambiguity/Uncertainty

- *"maybe"*, *"could be"*, *"sort of"*, *"kind of"*

3. Extremity

- *"very"*, *"extremely"*, *"super"*

4. Certainty/Exactness

- *"perfectly"*, *"completely"*, *"definitely"*

5. Neutrality

- *"medium"*, *"moderately"*, *"fairly"*

6. Negation

- *"not"*

Compound/holistic modifiers should be annotated jointly, e.g. if they modify each other.

- *very slightly*

- *not completely*

Morphological nuances should be captured at the token level, e.g.

- *darkish*, *med-dark*

Be careful that we do not annotate modifiers of spatial attributes inside markables.

### General Rules

Each expression should be annotated (i) *at the token level* and (ii) *within a single utterance*. When non-contiguous constructions need to be annotated, you can use the **split** option (make sure split is attached from the latter to the previous expressions).

- *"towards"* its *"left"* (split) ≈ towards its *"left"*

Predicative/generic markables should be ignored (deleted if necessary). Expressions that do not contain specific spatial information do not need to be annotated, e.g.

Non-specific or non-spatial information:

- Are they large or small?

- We have different views, so our dots are in different locations.

- Yes we found the same one.

When spans are difficult to annotate, make the best effort to identify the constructions that contain spatial (or modificational) information, e.g.

- All *"dark"* except one dot. (?) All *"dark except"* one dot.

- *"From left to right"*, there are ...

Finally, annotators should check the player's observation, raw dialogues and reference information to understand the context of the dialogue.