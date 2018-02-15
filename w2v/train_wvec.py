import json, re
import numpy as np
import nltk
from nltk import ngrams, word_tokenize, sent_tokenize, pos_tag
import gensim, logging
import os
import sys
import collections
from gensim import models
from gensim.models import Phrases
from gensim.models.keyedvectors import KeyedVectors
annotypes = ['Participants', 'Intervention', 'Outcome']
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
    sentences = load_sentences('/nlp/data/romap/ambig/definition_sents/')
    print len(sentences)
    model = gensim.models.Word2Vec(sentences, size=300)
    model.wv.save_word2vec_format('/nlp/data/romap/ambig/w2v/w2v100-300.txt', binary=False)



        
if __name__ == '__main__':
    train_unigram()
    word_sim()
