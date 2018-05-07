import json, re, os, sys, nltk
import numpy as np
from nltk import ngrams, word_tokenize, sent_tokenize, pos_tag
import gensim, logging, collections
from gensim import models
from gensim.models import Phrases
from gensim.models.keyedvectors import KeyedVectors
#path = '/Users/romapatel/Desktop/set/'

def tokenize(s):
    """
    :param s: string of the abstract
    :return: list of word with original positions
    """
    def white_char(c):
        return c.isspace() or c in [',', '?']
    res = []
    i = 0
    while i < len(s):
        while i < len(s) and white_char(s[i]): i += 1
        l = i
        while i < len(s) and (not white_char(s[i])): i += 1
        r = i
        if s[r-1] == '.':       # consider . a token
            res.append( (s[l:r-1], l, r-1) )
            res.append( (s[r-1:r], r-1, r) )
        else:
            res.append((s[l:r], l, r))
    return res

def load_sentences(dirname):
    doc_count, fin_sentences = 0, []

    definitions = os.listdir(dirname)
    for definition in definitions:
        for fname in os.listdir(dirname + definition + '/'):
            filepath = dirname + definition + '/' + fname
            f = open(filepath, 'r')
            for line in f:
                flag = False
                while True:
                    try:
                        sentence = line.decode('ascii').encode('ascii')
                        break
                    except UnicodeDecodeError: flag = True
                    except UnicodeEncodeError: flag = True
                    break
                if flag: continue
                sents = sent_tokenize(sentence)
                for sent in sents:
                    if definition not in sent: continue
                    words = word_tokenize(sent)
                    fin_sentences.append(words)
                        
    return fin_sentences



def train_unigram():
    
    f = open('/nlp/data/romap/law/title_sentences.json', 'r')
    sent_dict, sentences, ngrams = {}, [], []
    for line in f:
        temp = json.loads(line)
        sent_dict[temp['title']] = temp
    
    f = open('/nlp/data/romap/law/bigram_defs.txt', 'r')
    #f = open('/Users/romapatel/Desktop/bigram_defs.txt', 'r')
    for line in f:
        ngrams.append(line.lower().strip())

    ngrams = list(set(ngrams))
    ngrams = ['_'.join(item for item in word.split(' ')) for word in ngrams]
    #bigrams = [word for word in ngrams if len(word.split('_')) == 2]
    #ngrams = [word for word in ngrams if word not in bigrams]

    f = open('/nlp/data/romap/law/bigram_def_set.txt', 'w+')
    for word in ngrams:
        f.write(word + '\n')
        
    word_count = 0
    for title in sent_dict:
        for section in sent_dict[title]:
            if section == 'title': continue
            for sent_id in sent_dict[title][section]:
                sentence = sent_dict[title][section][sent_id]
                sentence = ' '.join(item for item in sentence)
                #replace bigrams
                for bigram in ngrams:
                    s = ' '.join(item for item in bigram.split('_'))
                    if re.search(s, sentence) is not None:
                        sentence = re.sub(s, bigram, sentence)
                #replace remaining ngrams
                '''
                for ngram in ngrams:
                    item = ' '.join(item for item in ngram.split('_'))
                    if re.search(item, sentence) is not None:
                        sentence = re.sub(item, ngram, sentence)
                '''
                words = sentence.split(' ')
                word_count += len(words)
                sentences.append(words)
                
    print len(sentences)
    model = gensim.models.Word2Vec(sentences, size=300)
    model.wv.save_word2vec_format('/nlp/data/romap/ambig/w2v/ngram-w2v100-300.txt', binary=False)



        
if __name__ == '__main__':
    train_unigram()
