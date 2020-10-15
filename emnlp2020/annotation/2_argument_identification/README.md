# Section 2. Argument Identification

In this section, we will explain how to annotate the arguments of spatial expressions (SEs), as well as modificants of the modifiers.

## Spatial Expression Arguments

We consider spatial expressions as **predicates** and annotate markables as their **arguments**. For simplicity, we only consider two types of arguments: subjects and objects. Spatial attributes can only have subjects as their arguments, and spatial relations can have both subjects and objects.

- [My dot] is *"dark"* (subject only)

- [It] is *"on the left"* of [another dot]. (subject and object)

- [Left dot] is *"the darkest"* in [the group]. (subject and object)

Spatial expressions can have multiple subjects/objects, as in 

- [Left dot] and [the right dots] are both *"dark"* (multiple subjects)

- ... is *"in between"* [large light dot] and [small black dot]. (multiple objects)

and no object need to be annotated in cases of nominal spatial relations (*"triangle"*), absolute relations (*"in the middle"*, *"on the left in my view"*), path expressions, and so on. If the judgements are ambiguous, you should mark them as **Ambiguous-Obj** and make the best choice possible (similary, you can use **Ambiguous-Subj** option).

No Object:

- [Left dot] is *"the darkest"*. (absolute relation)

- [Left dot] is *"darker in color"*. (absolute relation)

- [Three dots] *"angling up to the right"* (path expression)

Arguments can also be in previous utterances if not available in the present utterance. In general, we make the following priorities when multiple candidates exist:

1. Markables in the current utterance

2. Markables in the closest previous utterance of the same speaker

3. Markables in the closest previous utterance of different speaker

4. Any other choice

Based on our approach, there are several cases where it is difficult or impossible to capture the precise predicate argument structure (PAS). This includes cases of missing subject/object markables or complex PAS:

- [Three dots], *"going from left"* [small], [medium], [large].

If subject/object markables are missing (possibly due to existing annotation mistake), annotator should mark it as **No-Subj-Markable** or **No-Obj-Markable**. In any other cases where it is impossible to identify arguments, mark them as **Unannotatable**.

- [Bottom two] are *"large"* and *"larger"*. (No-Subj-Markable)

## Modificants

Modifiers can only have SEs as the modificants, and multiple modificants can be annotated if they exist. Candidate modificants should be annotated in the same priority as SE arguments (current utterance, previous same speaker utterance, previous different speaker utterance).

## Annotation Tips

When annotating the objects, we do not distinguish markables that include/exclude subject entities. Therefore, in the following example

-  I have [three dots], [two] dark and [one] light. The *"lighter"* dot ...

the objects of *"lighter"* can be either [three dots] or [two].

To reduce the annotation effort, we assume that 

(i) the default subject is the **last markable that starts before (or at the same time as) the spatial expression**

(ii) the default object is the **first markable that starts after the spatial expression**

(iii) the default modificant is the **last spatial expression that lasts after (or at the same time as) the modifier**

For example, following cases have default arguments (so do not need explicit annotation)

- [Two dots] are *"on the left of"* [the black dot].

- [*"Left"* dot] is darker than the [right dot].

- It is *"on the {very} edge"*.

and following cases have non-default arguments (so requires explicit annotation):

- *"On the left of"* [the two dots] is [a black dot].

- [Right dot] is lighter than the [*"Left"* dot].

- It is *"darker"* but {only slightly}.

