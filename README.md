# Find One In Common!

Repository for [A Natural Language Corpus of Common Grounding under Continuous and Partially-Observable Context](https://aaai.org/ojs/index.php/AAAI/article/view/4694) (Udagawa et al., AAAI 2019)

If you want to use our scripts/data in your research, please cite:

```
@inproceedings{udagawa2019natural,
  title={A Natural Language Corpus of Common Grounding under Continuous and Partially-Observable Context},
  author={Udagawa, Takuma and Aizawa, Akiko},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={7120--7127},
  year={2019}
}
```

# Generating Scenarios

To create 100 scenarios with number of shared entities = 4, 5 and 6 each, run

```
python generate_scenarios.py --num_world_each 100 --min_shared 4 --max_shared 6
```

# Dataset Analysis

The results of the paper can be reproduced by running

```
python simple_analysis.py --basic_statistics --count_dict --plot_selection_bias
```
# Experiments

Most of our experimental scripts are based on Facebook's [end-to-end-negotiator](https://github.com/facebookresearch/end-to-end-negotiator).

To conform our data to their format, run

```
python transform_to_txt.py --normalize
```

You can also create additional testsets by running

```
python transform_to_txt.py --normalize --uncorrelated
python transform_to_txt.py --normalize --success_only
```

In the `experiments` directory, you can train and test models with various configurations, e.g.

```
python train.py \
  --bsz 16 \
  --clip 0.1 \
  --dropout 0.5 \
  --init_range 0.01 \
  --lr 0.001 \
  --max_epoch 30 \
  --nembed_word 128 \
  --nembed_ctx 128 \
  --nhid_lang 128 \
  --nhid_sel 128 \
  --rel_ctx_encoder \
  --rel_hidden 128 \
  --seed 1
```

# Results

We report the results of each model with default configurations.

After major refactoring, numbers are slightly different but comparable to the ones reported in the paper.

|| Full | Uncorrelated | Success Only |
----|----|----|----
| Context Only (MLP) | 31.13 (std 0.6) | 31.56 | 34.26 |
| Context Only (RN) | 33.46 (std 0.9) | 32.44 | 35.00 |
| Context + Dialgoue (MLP) | 35.85 (std 1.6) | 38.07 | 38.89 |
| Context + Dialgoue (RN) | 41.50 (std 1.3) | 40.15 | 42.22 |

# Dialogue Interface

Available at `onecommon` directory.
