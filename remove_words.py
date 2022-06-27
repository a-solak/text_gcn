from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str, clean_str_twitter, spellcheck, loadWord2Vec
import sys
import time

strt = time.time()

if len(sys.argv) != 2:
    sys.exit("Use: python remove_words.py <dataset>")

datasets = ['twitter', 'twitter_large', '20ng', 'R8', 'R52', 'ohsumed', 'mr']
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("wrong dataset name")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# print(stop_words)

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.200d.txt'
# vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])
# dataset = '20ng'

doc_content_list = []
f = open('data/corpus/' + dataset + '.txt', 'rb')
# f = open('data/wiki_long_abstracts_en_text.txt', 'r')
for line in f.readlines():
    doc_content_list.append(line.strip().decode('latin1'))
f.close()

train_docs = []
pos_docs = []
neg_docs = []

# train docs for positive and negative tweets
f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
for i, line in enumerate(lines):
    temp = line.split("\t")
    if temp[1].find('train') != -1:
        train_docs.append(i)
        # if temp[2].find('positive') != 1:
        if temp[2] == 'positive':
            pos_docs.append(i)
        # elif temp[2].find('negative') != 1:
        elif temp[2] == 'negative':
            neg_docs.append(i)

print('made pos and neg doc number lists!')

word_freq = {}  # to remove rare words

# for twitter usage specifically
pos_word_freq = {}
neg_word_freq = {}

pos_upper_limit = 90000
neg_upper_limit = 180000

# is user url
def is_user_url(word):
    return word in ['<user>', '<url>', 'user', 'url']

for i, doc_content in enumerate(doc_content_list):
    if dataset == 'twitter' or dataset == 'twitter_large':
        temp = clean_str_twitter(doc_content)
    else:
        temp = clean_str(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
        # check if in pos docs or neg docs
        if i < pos_upper_limit:  # in pos_docs:
            if word in pos_word_freq:
                pos_word_freq[word] += 1
            else:
                pos_word_freq[word] = 1
        elif pos_upper_limit <= i < neg_upper_limit:  # in neg_docs:
            if word in neg_word_freq:
                neg_word_freq[word] += 1
            else:
                neg_word_freq[word] = 1

print('made pos and neg frequency lists!')

'''
print('********************\nPOSITIVE\n********************')

for k, val in pos_word_freq.items():
    if val > 50:
        print(k, val)

print('\n\n********************\nNEGATIVE\n********************')

for k, val in neg_word_freq.items():
    if val > 50:
        print(k, val)
'''

'''
for k, val in word_freq.items():
    if 2 <= word_freq[k] < 5:
        print(k, val)
'''

print('total time for building freq lists: ', time.time()-strt)

def common_interlabel(word, p_freq, n_freq, upper, lower):
    if word in p_freq and word in n_freq:
        if upper > (float(p_freq[word])/float(n_freq[word])) > lower:
            return True
    return False

'''
-------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------

PREPROCESSING CAN BE DONE HERE

-------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------
'''
empties = 0
increased_shorts = 0
combined_words = []
intercommons = []
intercommons_dict = {}

corrections = []

clean_docs = []
for i, doc_content in enumerate(doc_content_list):
    if dataset == 'twitter' or dataset == 'twitter_large':
        temp = clean_str_twitter(doc_content)
    else:
        temp = clean_str(doc_content)
    words = temp.split()
    doc_words = []
    for j, word in enumerate(words):
        # word not in stop_words and word_freq[word] >= 5
        if dataset == 'mr':  # or dataset == 'twitter':
            doc_words.append(word)
        elif dataset == 'twitter' or dataset == 'twitter_large':
            # for twitter dataset only take frequent, not user/url and not digit words and do lowercasing
            # preprocessing for twitter dataset
            # words themselves can be modified, for example all non alphanumeric symbols in the word could be replaced
            # with an empty string "". this could be done with regex
            if (not is_user_url(word)) and (not word.isdigit()):
                if word_freq[word] >= 5:
                    if (not common_interlabel(word, pos_word_freq, neg_word_freq, 1.05, 0.95)):
                        doc_words.append(word)
                        '''
                        if len(words) > j+1:
                            combined_word = word + words[j+1]
                            if combined_word in word_freq and word_freq[combined_word] >= 1000 and\
                                    (not common_interlabel(words[j+1], pos_word_freq, neg_word_freq, 1.05, 0.95)):
                                doc_words.append(combined_word)
                                combined_words.append(combined_word)
                        '''
                    else:
                        intercommons.append(word)
                        if word in intercommons_dict:
                            intercommons_dict[word] += 1
                        else:
                            intercommons_dict[word] = 1
                else:
                    word_candidates = spellcheck(word, word_freq, 5, 7)
                    if len(word_candidates) >= 1:
                        for w in word_candidates:
                            if (not is_user_url(w)) and (not w.isdigit()) and\
                                    (not common_interlabel(w, pos_word_freq, neg_word_freq, 1.05, 0.95)):
                                doc_words.append(w)
                                corrections.append((word, w))
        elif word not in stop_words and word_freq[word] >= 5:
            doc_words.append(word)

    if len(doc_words) < 1:
        if i < pos_upper_limit:
            doc_words.append('happy')
        elif pos_upper_limit <= i < neg_upper_limit:
            doc_words.append('sad')
        else:
            doc_words.append(words[0])
            empties += 1
    if (dataset == 'twitter' or dataset == 'twitter_large') and len(doc_words) < 5:
        doc_words = doc_words + doc_words
        increased_shorts += 1
    doc_str = ' '.join(doc_words).strip()
    #if doc_str == '':
        #doc_str = temp
    clean_docs.append(doc_str)

# for c in corrections:
#     print(c)
print('--> total corrected words: ', len(corrections))
print('total time: ', time.time()-strt)

print('--> words common in both positive and neg tweets: - \ntotal intercommons: ', len(intercommons),
      '\ntotal intercommons dict len: ', len(intercommons_dict.values()), '\ntotal pos words: ',
      len(pos_word_freq.values()), '\ntotal neg words: ', len(neg_word_freq.values()))
print('--> total of '+str(empties)+' empty words.')
print('--> total of '+str(increased_shorts)+' short words that were doubled.')
print('combined words: \n', combined_words)
print('--> total of '+str(len(combined_words))+' combined words.')

clean_corpus_str = '\n'.join(clean_docs)

f = open('data/corpus/' + dataset + '.clean.txt', 'w')
#f = open('data/wiki_long_abstracts_en_text.clean.txt', 'w')
f.write(clean_corpus_str)
f.close()

#dataset = '20ng'
min_len = 10000
aver_len = 0
max_len = 0 

f = open('data/corpus/' + dataset + '.clean.txt', 'r')
#f = open('data/wiki_long_abstracts_en_text.txt', 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    temp = line.split()
    aver_len = aver_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
    if len(temp) > max_len:
        max_len = len(temp)
f.close()
aver_len = 1.0 * aver_len / len(lines)
print('min_len : ' + str(min_len))
print('max_len : ' + str(max_len))
print('average_len : ' + str(aver_len))

print('total time: ', time.time()-strt)
