# Weakly-Supervised Aspect-Based Sentiment Analysis via Joint Aspect-Sentiment Topic Embedding 

The code and data used for our EMNLP paper Weakly-Supervised Aspect-Based Sentiment Analysis via Joint Aspect-Sentiment Topic Embedding.

## Requirements

* GCC compiler (used to compile the source c file): See the [guide for installing GCC](https://gcc.gnu.org/wiki/InstallingGCC).

## Datasets

We collect in-domain corpus for embedding training. For evaluation, we use Restaurant and Laptop datasets in [Sem-Eval 2015](http://alt.qcri.org/semeval2015/task12/) and [Sem-Eval 2016]()http://alt.qcri.org/semeval2016/task5/. We preprocessed these datasets in this repository.

## Run the Code

### Using the same datasets as ours
```
bash run_jasen.sh
```
This step runs the whole pipeline from embedding training, to neural network distillation and evaluation. The ``--dataset`` in the script is used to specify which prepared dataset (restaurant or laptop) to use. Each line starts with a parent node (with the root node being ROOT), and then followed by a ``tab``. The children nodes of this parent is appended and separated by ``space``. Generated embedding file is stored under ``${dataset}``.
Prediction results for each dataset are generated at ``/datasets/${dataset}/prediction.txt``.

### Preparing your own dataset
Create a new folder under ``/datasets`` for your new dataset. The in-domain unlabeled training corpus ``train.txt`` used for joint topic embedding training has the format of each line being a document. The test set ``test.txt`` used for evaluation is in following format:
```
line_id	aspect_label_id	sentiment_label_id	text
```
The keywords for each aspect and sentiment should be listed in ``aspect_w_kw.txt`` and ``senti_w_kw.txt``. Each line refers to one aspect/sentiment category. The line order should be consistent with the order of aspect and sentiment label ids. Examples can be found in prepared dataset folders.



