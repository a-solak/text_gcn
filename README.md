# text_gcn

The implementation of Text GCN in the paper:

Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377

Some changes made for better use with twitter data.

## Require

Python 2.7 or 3.6

Tensorflow >= 1.4.0

## Reproducing Results

1. Run `python remove_words.py twitter`

2. Run `python build_graph.py twitter`

3. Run `python train.py twitter`

4. Change `twitter` in above 3 command lines to 'twitter large'. Other possible datasets are: '20ng', 'R8', 'R52', 'ohsumed' and 'mr' when producing results for other datasets.

## Preprocessing

Preprocessing can be done in 'remove_words.py' and some preprocessing functions such as spellchecking are included in 'utils.py'.
