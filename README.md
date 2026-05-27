# Traffic data and graph neural networks
### Project 1 in INF367A : Topological Deep Learning
**Odin Hoff Gardå, March 2023**

![Toad](docs/toad.png)

# Introduction

We train and compare four machine learning models, one fully connected neural network and three graph neural networks. The objective is to predict traffic volumes for all traffic stations (nodes) for the next hour given the current traffic volumes, month, weekday and hour.

## Quick start

1. Run `unpack_data.py` to unpack compressed data file.
2. Run `preprocess_data.py` to pre-process data.
3. Run `create_data_summary.py` to generate a table summarizing the dataset (optional).
4. Run `train_and_evaluate.py` to train and evaluate any of the four models.

Configurations for pre-processing, training and the different models can be set in `config.py`. Plots of training and validation losses are saved in `figs/`. Plots of predictions and ground truth for some selected traffic stations are saved in `figs/<model name>_predictions/`. The data summary table is saved in `docs/` as a Markdown file.

