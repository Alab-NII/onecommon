# A Linguistic Analysis of Visually Grounded Dialogues based on Spatial Expressions

Repository for [A Linguistic Analysis of Visually Grounded Dialogues based on Spatial Expressions](https://arxiv.org/abs/2010.03127) (Udagawa et al., Findings of EMNLP 2020)

## Preparation

Install python packages using `requirements.txt`.

```
pip install -r requirements.txt
```

## Annotation

You can collect and analyze annotation in [data](https://github.com/Alab-NII/onecommon/tree/master/emnlp2020/data) directory.

First, we use the [annotated](https://github.com/Alab-NII/onecommon/tree/master/aaai2020/annotation/annotated) dialogues from [aaai2020](https://github.com/Alab-NII/onecommon/tree/master/aaai2020) to output dialogues in [brat annotatable format](https://brat.nlplab.org). To do this, run

```
python spatial_analysis.py --output_brat_format
```

You can annotate spatial expressions in the dialogue using our [annotation guideline](https://github.com/Alab-NII/onecommon/tree/master/emnlp2020/annotation). After collecting annotation, you can output the results in json format by running

```
python spatial_analysis.py --output_spatial_annotation
```

The output file will be in the following format:

```
spatial_annotation.json
|--chat_id
|  |--relations
|     |--[i]
|        |--start
|        |--end
|        |--text
|        |--subjects
|        |--objects
|        |--modifiers
|           |--[j]
|           |--start
|              |--end
|              |--text
|           |--canonical-function
|        |--tags
|        |--paraphrase
|        |--canonical-relations
|        |--splits
|  |--attributes
|     |--[i]
|        |--start
|        |--end
|        |--text
|        |--subjects
|        |--modifiers
|           |--[j]
|              |--start
|              |--end
|              |--text
|           |--canonical-function
|        |--tags
```

To test the reliability of the annotation, output results by different annotators (annotator_1 and annotator_2)

```
python spatial_analysis.py --output_spatial_annotation --annotator annotator_1
python spatial_analysis.py --output_spatial_annotation --annotator annotator_2
```

and then run

```
python reliability_analysis.py --span_agreement
python reliability_analysis.py --argument_agreement
python reliability_analysis.py --canonical_agreement
```

To compute basic statistics of the annotation, run

```
python spatial_analysis.py --compute_basic_statistics
```

## Reference Resolution

To train the models described in the paper, move to the [src](https://github.com/Alab-NII/onecommon/tree/master/emnlp2020/src) directory and run the following scripts:

```
train_num_ref_model.sh
train_num_ref_no_loc_model.sh
train_num_ref_no_size_model.sh
train_num_ref_no_color_model.sh
train_num_ref_no_size_color_model.sh
train_ref_model.sh
train_ref_no_loc_model.sh
train_ref_no_size_model.sh
train_ref_no_color_model.sh  
train_ref_no_size_color_model.sh
```

To test the models and output referent predictions, run

```
python test_reference.py --cuda --model_file num_ref_model --repeat_test
python test_reference.py --cuda --model_file num_ref_no_loc_model --repeat_test
python test_reference.py --cuda --model_file num_ref_no_size_model --repeat_test
python test_reference.py --cuda --model_file num_ref_no_color_model --repeat_test
python test_reference.py --cuda --model_file num_ref_no_size_color_model --repeat_test
python test_reference.py --cuda --model_file ref_model --repeat_test
python test_reference.py --cuda --model_file ref_no_loc_model --repeat_test
python test_reference.py --cuda --model_file ref_no_size_model --repeat_test
python test_reference.py --cuda --model_file ref_no_color_model --repeat_test
python test_reference.py --cuda --model_file ref_no_size_color_model --repeat_test
```

This will output referent predictions in json format, e.g. `num_ref_model_referent_annotation.json`.


## Model Analysis

You can conduct further model analyses using the scripts in [data](https://github.com/Alab-NII/onecommon/tree/master/emnlp2020/data) directory.

To plot & compare referent colors, run

```
python reference_analysis.py --plot_referent_color --referent_annotation aggregated_referent_annotation.json
python reference_analysis.py --plot_referent_color --referent_annotation num_ref_model_referent_annotation.json
```

To plot & compare referent sizes, run

```
python reference_analysis.py --plot_referent_size --referent_annotation aggregated_referent_annotation.json
python reference_analysis.py --plot_referent_size --referent_annotation num_ref_model_referent_annotation.json
```

You can test whether the referent predictions satisfy each canonical relation by running:

```
python spatial_analysis.py --test_canonical_relations
python spatial_analysis.py --test_all_models
```