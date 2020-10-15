# Annotation Guideline for Spatial Expressions

## Goal

The goal of this annotation project is to conduct a simple, useful and reliable annotation of **spatial expressions**, which play a central role in visual dialogues.

Following the definition of pustejovsky et al. (2011), we consider spatial expressions as  *"constructions that make explicit reference to the spatial attributes of an object or spatial relations between objects"*. We assume that referring expressions of physical objects (which we call *markables*) are already annotated.

The annotation procedure includes the following three steps:

1. **Span detection**

We detect the span of spatial expressions (spatial attributes and relations) and basic modifiers. [Check guideline](https://github.com/Alab-NII/onecommon/tree/master/emnlp2020/annotation/1_span_detection).

2. **Argument identification**

We identify the arguments of spatial expressions and the modificants of each modifier. [Check guideline](https://github.com/Alab-NII/onecommon/tree/master/emnlp2020/annotation/2_argument_identification).

3. **Canonicalization**

We annotate the **canonical relations** of spatial relations and **canonical functions** of modifiers. [Check guideline](https://github.com/Alab-NII/onecommon/tree/master/emnlp2020/annotation/3_canonicalization).