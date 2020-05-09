# TabNet Modified

Most of the code is taken from [here](https://github.com/google-research/google-research/tree/master/tabnet) for "TabNet: Attentive Interpretable Tabular Learning" by Sercan O. Arik and Tomas Pfister (paper: https://arxiv.org/abs/1908.07442).

The modified model, reduced TabNet, is defined in `model/tabnet_reduced.py`. There are two modifications:
* there is now 1 shared feature transformer and 1 decision-dependent feature transformer (from 2 and 2 before respectively), and
* the SparseMax mask for feature selection has been replaced by EntMax 1.5 (implementation in TensorFlow from [here](https://gist.github.com/justheuristic/60167e77a95221586be315ae527c3cbd)).

The combination of these modifications has improved the performance of TabNet with fewer parameters, particularly with a sharper mask for feature selection.

## Training and Evaluation

As in the original repository, this repository contains an example implementation of TabNet on the Forest Covertype dataset (https://archive.ics.uci.edu/ml/datasets/covertype). 

First, run `python download_prepare_covertype.py` to download and prepare the Forest Covertype dataset.
This command creates `train.csv`, `val.csv`, and `test.csv` files under the `data/` directory (will create the directory if it does not exist).

To run the pipeline for training and evaluation, simply use `python train_classifier.py`. Note that Tensorboard logs are written in `tflog/`.

To modify the experiment to other tabular datasets:
- Substitute the `train.csv`, `val.csv`, and `test.csv` files under `data/` directory,
- Create a new config in `config/` by copying `config/covertype.py` for the numerical and categorical features of the new dataset and hyperparameters,
- Reoptimize the TabNet hyperparameters for the new dataset in your config,
- Import the parameters in `train_classifier.py`,
- Select the modified TabNet architecture by setting `REDUCED = True`.
