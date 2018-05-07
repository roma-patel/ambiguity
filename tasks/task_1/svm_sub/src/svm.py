from __future__ import division
import os, math, operator, time, sys
from collections import defaultdict, namedtuple, Counter
import json, nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist, sent_tokenize, word_tokenize
from nltk import bigrams, trigrams, pos_tag
import numpy as np
import re, subprocess, itertools
from sklearn import svm, datasets
from sklearn.svm import LinearSVC
import pickle
#from sklearn.cross_validation import KFold, cross_val_score
#from sklearn.cross_validation import StratifiedKFold

path = '/nlp/data/romap/law/task_1/'

#set_name = train/meta/positive/ etc.
def createTitleFeat(set_names, cases):
    all_feat = []

    allcounts = 0
    for i in range(len(cases)):
        folder_path = path + '/data/' + set_names[i]  + '/'; case = cases[i]
        titles = [0.0 for i in range(1, 54)]; count = 0
        f = open(folder_path + case + '.json', 'r')
        for line in f:
            dict = json.loads(line)
        for title in dict['sections']:
            if title.isdigit() is False or int(title) > 54: continue
            if int(title) >= len(titles): continue
            titles[int(title)] += len(dict['sections'][title])
            count += len(dict['sections'][title])
        for i in range(len(titles)):
            if count == 0: continue
            titles[i] = (1.0*titles[i])/count
        all_feat.append(titles)
        
    return all_feat

#set_name = train/words/positive/, test/words/negative/ etc.
def createUnigramFeat(set_paths, cases, word_vecs):
    all_feat, names = [], []; 
    f = open(path + '/data/names.txt', 'r')
    for line in f: names.append(line.strip())
   
    for i in range(len(cases)):
        folder_path = path + '/data/' + set_paths[i]  + '/'; case = cases[i]
        count = 0; words = [0 for i in range(0, 300)]
        f = open(folder_path + case + '.txt', 'r')
        for line in f:
            word = line.strip()
            if word not in names and word in word_vecs.keys():
                words = [words[i]+word_vecs[word][i] for i in range(len(words))]
                count += 1
            if count > 100: break

        for i in range(len(words)):
            if count == 0: continue
            words[i] = (1.0*words[i])/count

        all_feat.append(words)
    return all_feat

#load all words from w2v, and for each case take top 100 words that are in w2v vocab
def load_bin_vec(fname):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            #if len(word_vecs.keys()) > 100: break
            word, flag = [], True
            while True:
                ch = f.read(1)
                if ord(ch) > 128: flag = False
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if flag is False or '_' in word or word.istitle():
                word = []
                continue
            arr = np.fromstring(f.read(binary_len), dtype='float32')
            word_vecs[word] = arr.tolist()

    print 'word vectors: ' + str(len(word_vecs.keys()))
    
    '''newf = open('/nlp/data/romap/law/task_1/svm/w2v_words.txt', 'w+')
    f = open('/nlp/data/romap/law/task_1/svm/w2v_words_sub.json', 'w+')
    f.write(json.dumps(word_vecs))
    for word in word_vecs: newf.write(word + '\n')'''
    
    return word_vecs

def load_wvec_dict():
    #f = open('/nlp/data/romap/law/task_1/svm/w2v_words.json', 'r')
    f = open('/nlp/data/romap/law/task_1/svm/w2v_words_sub.json', 'r')

    for line in f:
        word_vecs = json.loads(line)
    return word_vecs


def prepare_features(X_train, y_train, X_test, y_test):
    word_vecs = load_bin_vec('/nlp/data/corpora/GoogleNews-vectors-negative300.bin')
    print 'Loaded word vectors!'
    train_word_paths, train_title_paths = [], []; test_word_paths, test_title_paths = [], []

    for i in range(len(y_train)):
        if y_train[i] == 1: train_word_paths.append('/train/words/positive/')
        else: train_word_paths.append('/train/words/negative/')
        
        if y_train[i] == 1: train_title_paths.append('/train/meta/positive/')
        else: train_title_paths.append('/train/meta/negative/')
    #here test is dev!
    for i in range(len(y_test)):
        if y_test[i] == 1: test_word_paths.append('/train/words/positive/')
        else: test_word_paths.append('/train/words/negative/')
        
        if y_test[i] == 1: test_title_paths.append('/train/meta/positive/')
        else: test_title_paths.append('/train/meta/negative/')

        
    #unigram features       
    try:
        fileObject = open(path + "svm_sub/features/feat/train_unigram_feat.p", "r")
        train_unigram_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_unigram_feat = createUnigramFeat(train_word_paths, X_train, word_vecs)
        fileObject = open(path + "svm_sub/features/feat/train_unigram_feat.p", "wb")
        pickle.dump(train_unigram_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(path + "svm_sub/features/feat/test_unigram_feat.p", "r")
        test_unigram_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_unigram_feat = createUnigramFeat(test_word_paths, X_test, word_vecs)
        fileObject = open(path + "svm_sub/features/feat/test_unigram_feat.p", "wb")
        pickle.dump(test_unigram_feat, fileObject)
        fileObject.close()

    print 'Unigram done!'
    #title features
    try:
        fileObject = open(path + "svm_sub/features/feat/train_title_feat.p", "r")
        train_title_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_title_feat = createTitleFeat(train_title_paths, X_train)
        fileObject = open(path + "svm_sub/features/feat/train_title_feat.p", "wb")
        pickle.dump(train_title_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(path + "svm_sub/features/feat/test_title_feat.p", "r")
        test_title_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_title_feat = createTitleFeat(test_title_paths, X_test)
        fileObject = open(path + "svm_sub/features/feat/test_title_feat.p", "wb")
        pickle.dump(test_title_feat, fileObject)
        fileObject.close()
    print 'Titles done!'


    print 'Stacking features!'
    all_train_feat = np.hstack(
        (train_unigram_feat, train_title_feat))
    all_test_feat = np.hstack(
        (test_unigram_feat, test_title_feat))
    print 'Running classifier!'
    run_classifier(all_train_feat, y_train, all_test_feat,
    path + 'svm_sub/results/test.txt')
    print 'Done!'
    #get_accuracy(X_test, y_test)

def run_classifier(X_train, y_train, X_test, predicted_labels_file):
    svc = svm.SVC(decision_function_shape = 'ovo')
    svc.fit(X_train, y_train)
    y_test = svc.predict(X_test)
    confusion = []
    with open(predicted_labels_file, "w") as f_out:
        for label in y_test.tolist():
            f_out.write(str(label) + "\n")

#return lists of caseids
def get_data_sets():
    X_train, y_train, X_test, y_test, X_dev, y_dev = [], [], [], [], [], []
    
    train = '/nlp/data/romap/law/task_1/data/train/words/positive/'
    fileids = PlaintextCorpusReader(train, '.*').fileids()
    for fileid in fileids:
        X_train.append(fileid.split('.txt')[0]); y_train.append(1)
    train = '/nlp/data/romap/law/task_1/data/train/words/negative/'
    fileids = PlaintextCorpusReader(train, '.*').fileids()
    for fileid in fileids:
        X_train.append(fileid.split('.txt')[0]); y_train.append(0)
    train_len = len(X_train)

    '''for i in range(len(X_train)-800, len(X_train)):
        X_dev.append(X_train[i]); y_dev.append(y_train[i])
    X_train = X_train[:-800]; y_train = y_train[:-800]'''

    #first 100 +ve and last 800 -ve
    X_dev = X_train[:100]; X_dev.extend(X_train[-800:])
    y_dev = y_train[:100]; y_dev.extend(y_train[-800:])

    X_train = X_train[100:]; y_train = y_train[100:]
    X_train = X_train[:-800]; y_train = y_train[0:-800]

    
    test = '/nlp/data/romap/law/task_1/data/test/words/positive/'
    fileids = PlaintextCorpusReader(test, '.*').fileids()
    for fileid in fileids:
        X_test.append(fileid.split('.txt')[0]); y_test.append(1)
    test = '/nlp/data/romap/law/task_1/data/test/words/negative/'
    fileids = PlaintextCorpusReader(test, '.*').fileids()
    for fileid in fileids:
        X_test.append(fileid.split('.txt')[0]); y_test.append(0)
        

    return [X_train, y_train], [X_test, y_test], [X_dev, y_dev]



if __name__ == '__main__':
    train, test, dev = get_data_sets()
    X_train = train[0]; y_train = train[1]; X_dev = dev[0]; y_dev = dev[1]
    
    X_train = X_train[:1000]; y_train = y_train[:1000]
    X_dev = X_dev[:200]; y_dev = y_dev[:200]

    print len(X_train)
    print len(X_dev)
    prepare_features(X_train, y_train, X_dev, y_dev)





