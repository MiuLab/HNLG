# Natural Language Generation by Hierarchical Decoding with Linguistic Patterns
## References
Main papers to be cited ([NAACL2018](https://arxiv.org/abs/1808.02747) and [SLT2018]()):

```
@inproceedings{su2018natural,
    title={Natural Language Generation by Hierarchical Decoding with Linguistic Patterns},
    author={Shang-Yu Su, Kai-Ling Lo, Yi-Ting Yeh, and Yun-Nung Chen},
    booktitle={Proceedings of The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
    year={2018}
}

@inproceedings{su2018investigating,
    title={Investigating Linguistic Pattern Ordering in Hierarchical Natural Language Generation},
    author={Shang-Yu Su, and Yun-Nung Chen},
    booktitle={Proceedings of The 7th IEEE Workshop on Spoken Language Technology},
    year={2018}
}
```

## Setup

```
# Get the E2ENLG dataset (from link below), and put it under data/E2ENLG/
$ mkdir -p data/E2ENLG/
# take conda for example, create a new environment
$ conda create -n [your_env_name] python=3
$ source activate [your_env_name]
$ conda install pytorch torchvision -c pytorch
$ conda install spacy nltk
# download the spaCy models
$ python -m spacy download en
```

## Usage

<b>Please refer to `example_train.sh` and `example_test.sh` for the examples of commands.</b>

```
# under src/
$ python3 train.py --data_dir=../data/
```

Optional Arguments:

```
# under src/
$ python3 train.py --help
```

## Data

### E2E NLG:
[Link](http://www.macs.hw.ac.uk/InteractionLab/E2E/)


