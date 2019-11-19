# An Annotated Corpus of Reference Resolution for Interpreting Common Grounding

Repository for "An Annotated Corpus of Reference Resolution for Interpreting Common Grounding" (Udagawa and Aizawa, AAAI 2020)

## Annotation Collection and Analysis

Annotation scripts are provided in [annotation]((https://github.com/Alab-NII/onecommon/tree/master/aaai2020/annotation) subdirectory.

First, we need to transform raw dialogues into [brat annotatable format]((https://brat.nlplab.org)).

In order to automatically detect (minimal) noun phrases, follow <https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK> and make sure Stanford CoreNLP can be used from nltk. We only use successful dialogues, and we also preprocess raw dialogues to fix obvious misspellings and grammatical errors. To do this, run

```
python reference_annotation.py --output_brat_format --success_only --correct_misspellings --replace_strings
```

After detecting markables based on our guidelines (`markable_detection.md`), convert it into json format to annotate referents based on our web interface:

```
python reference_annotation.py --output_markable_annotation
```

Based on the collected referent annotation, you can create gold annotation through label aggregation (i.e. majority voting at the entity level):

```
python reference_annotation.py --referent_aggregation
```

To compute agreement statistics, run 

```
python reference_annotation.py --markable_agreement --batch_id batch_33
python reference_annotation.py --referent_agreement
```

For more details on disagreements, run 

```
python reference_annotation.py --referent_disagreement
```

Finally, to reproduce our analysis of the pragmatic expressions of color, run

```
python reference_annotation.py --referent_color
```

## Data Format

Annotation of markables are structured as follows:

```
markable_annotation.json
|--chat_id
|  |--markables
|     |--[i]
|        |--markable_id
|        |--start
|        |--end
|        |--all-referents
|        |--no-referent
|        |--generic
|        |--predicative
|        |--anaphora
|        |--cataphora
|        |--speaker
|        |--text (of the markable)
|  |--text (of the dialogue)

```

Annotation of the referents are structured as follows:

```
referent_annotation.json
|--chat_id
|  |--annotator
|     |--markable_id
|        |--referents
|           |--[i]
|              |--entity_id
|        |--ambiguous
|        |--unidentifiable
|        |--assignment_id
```


## Experiments

To generate data for experiments, go to `annotation` directory and run

```
python transform_referents_to_txt.py --normalize
python transform_scenarios_to_txt.py --normalize --input_file shared_4.json --output_file shared_4.txt
python transform_scenarios_to_txt.py --normalize --input_file shared_5.json --output_file shared_5.txt
python transform_scenarios_to_txt.py --normalize --input_file shared_6.json --output_file shared_6.txt
```

Other experimental scripts are provided in `experiments` directory.

To train the models described in the paper, use `train_tsel_model.sh`, `train_ref_model.sh`, `train_tsel_ref_model.sh`, `train_tsel_dial_model.sh` and `train_tsel_ref_dial_model.sh`.

To test the models on target selection and reference resolution tasks, run

```
python test_reference.py --cuda --model_file tsel_model --cuda --repeat_test
python test_reference.py --cuda --model_file ref_model --cuda --repeat_test
python test_reference.py --cuda --model_file tsel_ref_model --cuda --repeat_test
python test_reference.py --cuda --model_file tsel_dial_model --cuda --repeat_test
python test_reference.py --cuda --model_file tsel_ref_dial_model --cuda --repeat_test
```

To test the models on selfplay dialogue tasks, run

```
python selfplay.py --alice_model_file tsel_dial_model --bob_model_file tsel_dial_model --temperature 0.25 --context_file shared_4 --cuda --repeat_selfplay
python selfplay.py --alice_model_file tsel_dial_model --bob_model_file tsel_dial_model --temperature 0.25 --context_file shared_5 --cuda --repeat_selfplay
python selfplay.py --alice_model_file tsel_dial_model --bob_model_file tsel_dial_model --temperature 0.25 --context_file shared_6 --cuda --repeat_selfplay
python selfplay.py --alice_model_file tsel_ref_dial_model --bob_model_file tsel_ref_dial_model --temperature 0.25 --context_file shared_4 --cuda --repeat_selfplay
python selfplay.py --alice_model_file tsel_ref_dial_model --bob_model_file tsel_ref_dial_model --temperature 0.25 --context_file shared_5 --cuda --repeat_selfplay
python selfplay.py --alice_model_file tsel_ref_dial_model --bob_model_file tsel_ref_dial_model --temperature 0.25 --context_file shared_6 --cuda --repeat_selfplay
```

Finally, to visualize the selfplay dialogues, you need to train the markable detector.

To do this, run 
```
python train_markables.py --bsz 1 --nhid_lang 256 --unk_threshold 10 --model_file markable_detector --seed 1
```
Pretrained model is also available (`markable_detector_1.th`)

Then, you should run
```
python selfplay.py --alice_model_file tsel_ref_dial_model --bob_model_file tsel_ref_dial_model --temperature 0.25 --context_file shared_5 --cuda --seed 1 --record_markables
```

This will create the files `selfplay_markables.json` and `selfplay_referents.json`, which can be used for visualization based on our [webapp](https://github.com/Alab-NII/onecommon/tree/master/webapp).