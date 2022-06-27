#!/usr/bin/python
#-*-coding:utf-8-*-

import random

random.seed(42)

'''
dataset_name = 'own'
sentences = ['Would you like a plain sweater or something else?​', 'Great. We have some very nice wool slacks over here. Would you like to take a look?']
labels = ['Yes' , 'No' ]
train_or_test_list = ['train', 'test']
'''

dataset_name = 'twitter'

n_data = []
p_data = []

# read positive and negative tweet data
# use own path here
with open('C:/Users/Ahmet/ETH_Master/FS 22/CIL/twitter_ds/twitter-datasets/train_neg.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    n_data = lines
print('read negative tweet lines ', len(n_data))

with open('C:/Users/Ahmet/ETH_Master/FS 22/CIL/twitter_ds/twitter-datasets/train_pos.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    p_data = lines
print('read positive tweet lines ', len(p_data))

random.shuffle(n_data)
random.shuffle(p_data)

# fill all tweet data and target labels
tweet_train_data = p_data[:90000] + n_data[:90000]
tweet_test_data = p_data[90000:] + n_data[90000:]
sentences = tweet_train_data + tweet_test_data
labels = ['positive' for _ in p_data[:90000]] + ['negative' for _ in n_data[:90000]] +\
         ['positive' for _ in p_data[90000:]] + ['negative' for _ in n_data[90000:]]
train_or_test_list = ['train' for _ in tweet_train_data] + ['test' for _ in tweet_test_data]

meta_data_list = []

for i in range(len(sentences)):
    meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
    meta_data_list.append(meta)

meta_data_str = '\n'.join(meta_data_list)

f = open('data/' + dataset_name + '.txt', 'w')
f.write(meta_data_str)
f.close()

corpus_str = '\n'.join(sentences)

f = open('data/corpus/' + dataset_name + '.txt', 'w')
f.write(corpus_str)
f.close()