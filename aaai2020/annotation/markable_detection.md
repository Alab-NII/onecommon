# Annotation Guideline for Markable Detection

## Goal

The goal of this annotation project is to provide **simple**, **useful** and **reliable** annotation of reference resolution, which includes the following two steps:

Step 1. Markable Detection

- standardized annotation of referring expressions

Step 2. Reference Resolution

- annotation of the referents of referring expressions

This guideline is on how to conduct **Markable Detection** (Step 1).

Resolving reference (including coreference/anaphoric chains) is a fundamental skill in common grounding, and we expect this project to play a critical role in understanding both human and machine common grounding strategies.

## Section 1. Markable Definition

A **markable** is a minimal, contiguous and non-overlapping text span that can be considered as a referring expression of the entities currently under consideration (in our case, the dots in the circular view).

Basically, we annotate a markable as a **minimal noun phrase** (noun phrase including prenominal modifiers but excluding all postnominal modifiers). Prenominal modifiers include the determiners (*"a"*, *"the"*, *"another"*), quantifiers (*"all of"*,  *"most of"*, *"a few of"*) and adjectives (*"large"*, *"dark"*, *"medium"*). Postnominal modifiers include prepositional phrases (*"to the left"*, *"of medium size"*), appositive phrases (*", the largest in my view, "*), and relational clauses (*"which is black"*).

### Quantifiers and the Preposition "of"

However, it could be ambiguous to annotate quantifiers because they include the preposition *"of"*, and we can consider *"two of the three"* as a single markable (*"two of"* is a numerical quantifier) or two different markables concatenated by a prepositional phrase (*"two"* and *"the three"* are different markables). This seems to be an almost genuinely ambiguous issue, so we allow annotation in either ways:

- *"two of the three"* = *"two"* of *"the three"*
- *"one of them"* = *"one"* of *"them"*
- *"none of those"* = *"none"* of *"those"*

If the following markable is **Generic** or **Predicative** (discussed later), do not annotate them as independent markables. Thus,

- *"the rest of my dots"* = *"the rest"* of my dots
- *"a close triangle of three darker dots"* = *"a close triangle"* of three darker dots

As a generalization of this rule, you may include postnominal modifiers starting with the preposition *"of"*, if they (i) do not change the reference of the markable, (ii) do not contain other markables, and (iii) improve readability and identification of the markable.

- *"a group of dots"* = *"a group"* of dots
- *"4 shades of grey"* = *"4 shades"* of grey
- *"a group of 3 dots"* = *"a group"* of 3 dots

### Possessives and Reflexives

We also consider **possessives** and **reflexives** as independent markables:

possessives:
- *"their"* left
- on *"its"* own

reflexives:
- by *"itself"*
- *"themselves"*

If a markable has possessives as their prenomial modifiers, we allow annotating them as either a single markable or different markables:

- *"Its neighbor"* = *"Its"* *"neighbor"*
- *"the triangle's top dot"* = *"the triangle"*'s *"top dot"*

We do not distinguish the annotation of possessives with or without clitics.

- **the dot's** color = **the dot**'s color
- **it's** next to = **it**'s next to

### Determiners

We find various types of determiners in our dataset, including (but not limited to) the following list:

definite: *the*

indefinite: *a*, *an*

demonstrative: *that*, *those*, *these*, *this*

possessives: *my*, *your*, *our*

others: *one*, *two*, *both*, *other*, *another*, *every*, *each*, *either*, *neither*, *any*, *no*

We discuss how to deal with specific types of determiners in Section 2.

### Adjectives

Note that adjectives should be included in the markable only if they precede the head of the noun phrase: *"a very dark dot"* is a markable (*"very dark"* is included), but the markable is *"a dot"* in *"a dot very dark"* (*"very dark"* is not included). However, this could also be ambiguous when the explict head is missing. For example, *"one very dark"* may be annotated as:

- *"one"* very dark (*"one dot"* very dark)
- *"one very dark"* (*"one very dark dot"*)

We allow annotating ambiguous cases in either ways.

### Nominal Head

Typical head of the noun phrase referring to the dots include (but not limited to) the following nouns/pronouns:

Nouns:

  - dot
  - dots
  - pair
  - triangle
  - group
  - cluster

Pronouns:

  - one
  - ones
  - it
  - they
  - them
  - itself

However, due to the free-formed nature of dialogues, explicit nominal heads are sometimes omitted: in such cases, the head of the markables could be adjectives, numerals, or quantifiers as in the following examples:

  - largest (*"the largest"* is ...)
  - dark (*"the very dark"* has ...)
  - black (*"large black"* to the bottom right ...)
  - all (*"all"* clustered ...)
  - two (*"two"* of which are ...)

In general, we will not annotate **Predicative Noun Phrases** as markables. However, if there is no predicated NP with the same denotation, we consider the Predicative NP as a new referring expression and an independent markable:

"a triangle" is not a markable (Predicative):
 - *"Three dots"* in a triangle
 - *"Three dots"* are forming a triangle
 - *"Three dots"* making a triangle

"a triangle" is a markable:
 - *"One dark dot"* and *"two light dots"* in *"a triangle"*

other examples of predicative NP:
- *"2 sets of dots"* like the one you described
- *"my darkest dot"* is also the largest
- *"that"* must be the common dot

## Section 2. Attributes of Markables

Sometimes, it is clear from the context that the markable's denotation is empty. For example, we can consider the following markables to have clearly empty denotations:

  - *"none"*
  - *"nothing"*
  - I don't have *"it"*

and sometimes it is more ambiguous:

  - not sure if i have *"one"* that's alone.
  - *"it"* may be of yours alone

If it is clear from the context that the denotation is empty, the markable should be annotated with the attribute **No-Referent**.

Similarly, if it is clear from the context that the denotation is all of the 7 dots in one's view, the markable should be annotated with the attribute **All-Referents**.

Sometimes, it is clear from the context that the markables is not specific enough to identify the referents. This usually includes:

Interrogatives:
  - *"What"* do you have around it?
  - *"which dot"* would you like to try for?
  - My smallest dot is the darkest. is *"yours"*?

Indefinites:
  - Do you see *"anything"* else?
  - if not describe *"something else"*
  - are there *"any dots"* around them?
  - Do you have *"any unique dots"*?

Markables with meta-information:
  - *"Our dots"* will not be in the same place
  - *"the dots"* are in different positions in our respective circles

Note that interrogatives and indefinites are not necessarily generic, as in the following examples:

Not generic:
  - I also have *"what"* you described
  - I see *"something"* similar on the left top of my circle as well.

If it is clear  from the context that the markable is not specific enough to identify the referents, it should be annotated with the attribute **Generic**.

## Section 3. Relations between Markables

To reduce the annotation effort, we annotate obvious **anaphoric/cataphoric links** within the same utterance if more than two markables have clearly identical denotations. If the anaphoric relation is ambiguous, you do not need to annotate them.

Examples (latter are anaphora of the former):
  - *"large dot"* by *"itself"*
  - *"a large black dot"* with a smaller dot to the left of *"it"*
  - *"two dots"* close to *"each other"*

## Section 4. Annotation Tips

Some nouns are ambiguous and may refer to the dots or something else: these need to be disambiguated based on the dialogue context.

  - circle (may refer to the dots or the outline of the view)
  - line (may refer to the dots or the outline of the view)
  - spot (may refer to the dots or the area of the view)
  - mine (may refer to the dots or the whole view)

We will only annotate explicit markables: thus, zero anaphora (and other types of ellipsis) do not need to be annotated.

  - smaller but same light grey
  - toward the edge or more toward the middle of the top?
  - to the left of the tiny left one
  - small and very dark? towards the bottom?

Sometimes pronouns can be used as formulaic expressions or refer to other entities (such as proposal or plan). If they are not specifically referring to the dots, these should not be annotated as markables:

- lets do it
- got it
- Yes, you got it!

However, if it is more natural to consider them as markables, you should annotate them as markables:

- Yes, I got *"it"*.
- I say we go for *"that"*

Free-formed dialogues also contain many spelling/grammatical mistakes. Many of them are corrected during preprocessing, but if you find **critical spelling/grammatical mistakes with in the span of a markable**, please fix them **if the correct spelling is predictable** (using the *notes* in brat):

Predictable
- Also *"am gray"* at top left --> Also *"a gray"* at top left

If it is not predictable, leave the errors as is and make the best effort to identify the markable. Also, **if the error is repaired in the dialogue**, you should not fix them, as in the following example:

0: I see one *"ark dot"*

1: What do you mean by ark?

0: I meant dark

Finally, it is always recommended to check each player's view when you find difficulty in annotating markables.
