from __future__ import division
import os, math, operator, time
from collections import defaultdict, namedtuple, Counter
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist, sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk import bigrams, trigrams
import numpy as np
import re, subprocess, itertools
import pickle
import sys
from sklearn import svm
from sklearn import datasets
from sklearn.svm import LinearSVC
import json
#from sklearn.cross_validation import KFold, cross_val_score
#from sklearn.cross_validation import StratifiedKFold
annotypes = ['Participants', 'Intervention', 'Outcome']
path = '/nlp/data/romap/sentence-classifier/'

#path = '/Users/romapatel/Desktop/sentence-classifier/'
set_path = 'sets/1/'

def createBigramFeat(sents):
    all_feat = []
    bigram_space = extract_bigrams()
    for sentence in sents:
        words = nltk.word_tokenize(sentence.lower())
        bigrams = nltk.bigrams(words)
        bigram_list = []
        for item in bigrams:
            bigram_list.append(' '.join(i for i in item))
        vector = map_bigrams(bigram_list, bigram_space)
        all_feat.append(vector)
    return all_feat

def map_bigrams(word_list, bigram_space):
    word_count = dict(Counter(word_list))
    output_vector = []
    for tag in bigram_space:
        if tag in word_count.keys():
            output_vector.append(word_count[tag] / len(word_list))
        else:
            output_vector.append(0)
    return output_vector

def extract_bigrams():
    bigram_list, stopwords = [], []
    f = open(path + 'features/bigrams.txt', 'r')
    for line in f:
        bigram_list.append(line.strip())
    return bigram_list

def extract_unigrams():
    unigram_list, stopwords = [], []
    f = open(path + 'features/unigrams.txt', 'r')
    for line in f:
        unigram_list.append(line.strip())       
    return unigram_list

def map_unigrams(word_list, unigram_space):
    word_count = dict(Counter(word_list))
    output_vector = []

    for tag in unigram_space:
        if tag in word_count.keys():
            output_vector.append(word_count[tag] / len(word_list))
        else:
            output_vector.append(0)
    return output_vector

def createUnigramFeat(sents):
    all_feat = []
    unigram_space = extract_unigrams()
    for sentence in sents:
        words = nltk.word_tokenize(sentence)
        vector = map_unigrams(words, unigram_space)
        all_feat.append(vector)
    return all_feat

def map(word_list, pos_list):
    fin_list, open_list = [], []
    f = open(path + 'features/prep.txt', 'r')
    for line in f:
        open_list.append(line.strip())
        
    for i in range(len(word_list)):
        if word_list[i] in open_list:
            fin_list.append(word_list[i])
        elif word_list[i] == '(':
            fin_list.append('-LLB-')
        elif word_list[i] == ')':
            fin_list.append('-RRB-')
        else:
            fin_list.append(pos_list[i])
    return fin_list

def extract_pos_patterns(train_list):
    pos_pattern_list = []
    f = open(path + 'features/pos_pattern_space.txt', 'r')
    for line in f:
        pos_pattern_list.append(line.strip())
    return sorted(list(set(pos_pattern_list)))

def map_pos_patterns(pos_pattern_list, pos_pattern_space):
    pos_count = dict(Counter(pos_pattern_list))
    output_vector = []
    
    for pattern in pos_pattern_space:
        if pattern in pos_count.keys():
            output_vector.append(pos_count[pattern] / len(pos_pattern_list))
        else:
            output_vector.append(0)
    return output_vector

def createPOSPatternFeat(sents):
    all_feat = []
    pos_pattern_space = extract_pos_patterns(sents)
    for sentence in sents:
        words = nltk.word_tokenize(sentence)
        tag_tuples = nltk.pos_tag(words)
        tags = [item[1] for item in tag_tuples]
        pattern_bigrams = nltk.bigrams(map(words, tags))
        bigrams = []
        for item in pattern_bigrams:
            bigrams.append(' '.join(i for i in item))
        vector = map_pos_patterns(bigrams, pos_pattern_space)
        all_feat.append(vector)
    return all_feat

def extract_pos_tags():
    pos_tag_list = []
    f = open(path + 'features/pos.txt', 'r')
    for line in f:
        pos_tag_list.append(line.strip())
       
    return sorted(list(set(pos_tag_list)))

def map_pos_tags(pos_list, pos_tag_space):
    pos_count = dict(Counter(pos_list))
    output_vector = []
    
    for tag in pos_tag_space:
        if tag in pos_count.keys():
            output_vector.append(pos_count[tag] / len(pos_list))
        else:
            output_vector.append(0)
    return output_vector

def createPOSFeat(sents):
    all_feat = []
    pos_tag_space = extract_pos_tags()
    
    for sentence in sents:
        words = nltk.word_tokenize(sentence)
        tag_tuples = nltk.pos_tag(words)
        tags = [item[1] for item in tag_tuples]
        vector = map_pos_tags(tags, pos_tag_space)
        all_feat.append(vector)
    return all_feat

def extract_aslog_bigrams():
    part_frames, int_frames, out_frames = [], [], []
    fin_frames = []
    f = open(path + 'features/participants_frames.tsv', 'r')
    for line in f:
        items = line.strip().split(' ')
        if len(items) <= 1: continue
        pattern = ' ' + line.strip() + ' '
        part_frames.append(pattern)
        fin_frames.append(pattern)

    f = open(path + 'features/intervention_frames.tsv', 'r')
    for line in f:
        items = line.strip().split(' ')
        if len(items) <= 1: continue
        pattern = ' ' + line.strip() + ' '
        int_frames.append(pattern)
        fin_frames.append(pattern)

    f = open(path + 'features/outcome_frames.tsv', 'r')
    for line in f:
        items = line.strip().split(' ')
        if len(items) <= 1: continue
        pattern = ' ' + line.strip() + ' '
        out_frames.append(pattern)
        fin_frames.append(pattern)

    return (list(set(fin_frames))), part_frames, int_frames, out_frames

def map_aslog_bigrams(bigrams, aslog_space):
    bigram_count = dict(Counter(bigrams))
    output_vector = []
    for bigram in aslog_space:
        bigram = bigram.strip()
        if bigram in bigram_count.keys():
            output_vector.append(bigram_count[bigram] / len(bigrams))
        else:
            output_vector.append(0)
    return output_vector

def createASlogFeat_2(sents):
    all_feat = []
    aslog_space, part_space, int_space, out_space = extract_aslog_bigrams()
    
    for sentence in sents:
        words = nltk.word_tokenize(sentence.lower())
        bigrams = nltk.bigrams((words))
        bigram_list = []
        for item in bigrams:
            bigram_list.append(' '.join(i for i in item))
        vector = map_aslog_bigrams(bigram_list, aslog_space)
        all_feat.append(vector)
    return all_feat

def createASlogFeat(sents):
    all_feat = []
    aslog_space, part_space, int_space, out_space = extract_aslog_bigrams()

    for sentence in sents:
        length = len(word_tokenize(sentence))
        vector = [0, 0, 0]
        for frame in part_space:
            if frame in sentence.lower(): vector[0] += 1
        for frame in int_space:
            if frame in sentence.lower(): vector[1] += 1
        for frame in out_space:
            if frame in sentence.lower(): vector[2] += 1
        norm_vector = [min(1.0*item/length, 1) for item in vector]
        all_feat.append(norm_vector)
    return all_feat

def extract_indwords():
    part_words, int_words, out_words = [], [], []
    f = open(path + 'features/participants_words.tsv', 'r')
    for line in f:
        items = line.strip().split('\t')
        word, prec = items[1], float(items[0])
        if prec < 0.8: continue
        if re.search('[0-9/\',.]', word) is not None: continue
        part_words.append(word)
    f = open(path + 'features/participants_words_amt.tsv', 'r')
    for line in f:
        items = line.strip().split('\t')
        word, prec = items[1], float(items[0])
        if prec < 0.8: continue
        if re.search('[0-9/\',.]', word) is not None: continue
        part_words.append(word)
    f = open(path + 'features/intervention_words.tsv', 'r')
    for line in f:
        items = line.strip().split('\t')
        word, prec = items[1], float(items[0])
        if prec < 0.8: continue
        if re.search('[0-9/\',.]', word) is not None: continue

        int_words.append(word)
    f = open(path + 'features/intervention_words_amt.tsv', 'r')
    for line in f:
        items = line.strip().split('\t')
        word, prec = items[1], float(items[0])
        if prec < 0.8: continue
        if re.search('[0-9/\',.]', word) is not None: continue

        int_words.append(word)
    f = open(path + 'features/outcome_words.tsv', 'r')
    for line in f:
        items = line.strip().split('\t')
        word, prec = items[1], float(items[0])
        if prec < 0.8: continue
        if re.search('[0-9/\',.]', word) is not None: continue

        out_words.append(word)
    f = open(path + 'features/outcome_words_amt.tsv', 'r')
    for line in f:
        items = line.strip().split('\t')
        word, prec = items[1], float(items[0])
        if prec < 0.8: continue
        if re.search('[0-9/\',.]', word) is not None: continue

        out_words.append(word)
    return part_words, int_words, out_words

def createIndWordFeat(sents):
    all_feat = []
    part_words, int_words, out_words = extract_indwords()
    for sentence in sents:
        #print sentence
        words = word_tokenize(sentence.lower())
        length = len(words)
        vector = [0, 0, 0]
        for frame in part_words:
            if frame in words:
                vector[0] += 1
        for frame in int_words:
            if frame in words:
                vector[1] += 1
        for frame in out_words:
            if frame in words:
                vector[2] += 1
        norm_vector = [min(1.0*item/length, 1) for item in vector]
        #print norm_vector
        all_feat.append(norm_vector)
    return all_feat

def extract_wordlists():
    part_words, int_words, out_words = [], [], []
    f = open(path + 'features/disease_names.txt', 'r')
    for line in f:
        word = line.strip()
        part_words.append(word)

    f = open(path + 'features/drug_names.txt', 'r')
    for line in f:
        word = line.strip()
        if len(word) <= 4: continue
        int_words.append(word)

    f = open(path + 'features/outcome_names.txt', 'r')
    for line in f:
        word = line.strip()
        if len(word) <= 4: continue
        out_words.append(word)
    return part_words, int_words, out_words

def createIndWordFeat_2(sents):
    all_feat = []
    diseases, drugs, outcomes = extract_indwords()
    for sentence in sents:
        #print sentence
        words = word_tokenize(sentence.lower())
        length = len(words)
        vector = [0, 0, 0]
        for frame in diseases:
            if frame in sentence:
                vector[0] += 1
        for frame in drugs:                
            if frame in sentence:
                vector[1] += 1
        for frame in outcomes:
            if frame in sentence:
                vector[2] += 1
        norm_vector = [min(1.0*item/length, 1) for item in vector]
        #print norm_vector
        all_feat.append(norm_vector)
    return all_feat

def prepare_features(X_train, y_train, X_test, y_test):
    #indword features type_2
    '''try:
        fileObject = open(path + "features/feat/train_indword_2_feat.p", "r")
        train_indword_2_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_indword_2_feat = createIndWordFeat_2(X_train)
        fileObject = open(path + "features/feat/train_indword_2_feat.p", "wb")
        pickle.dump(train_indword_2_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(path + "features/feat/test_indword_2_feat.p", "r")
        test_indword_2_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_indword_2_feat = createIndWordFeat_2(X_test)
        fileObject = open(path + "features/feat/test_indword_2_feat.p", "wb")
        pickle.dump(test_indword_2_feat, fileObject)
        fileObject.close()'''

    #indword features type_1
    try:
        fileObject = open(path + "features/feat/train_indword_feat.p", "r")
        train_indword_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_indword_feat = createIndWordFeat(X_train)
        fileObject = open(path + "features/feat/train_indword_feat.p", "wb")
        pickle.dump(train_indword_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(path + "features/feat/test_indword_feat.p", "r")
        test_indword_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_indword_feat = createIndWordFeat(X_test)
        fileObject = open(path + "features/feat/test_indword_feat.p", "wb")
        pickle.dump(test_indword_feat, fileObject)
        fileObject.close()
        
    #POS features
    try:
        fileObject = open(path + "features/feat/train_pos_feat.p", "r")
        train_pos_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_pos_feat = createPOSFeat(X_train)
        fileObject = open(path + "features/feat/train_pos_feat.p", "wb")
        pickle.dump(train_pos_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(path + "features/feat/test_pos_feat.p", "r")
        test_pos_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_pos_feat = createPOSFeat(X_test)
        fileObject = open(path + "features/feat/test_pos_feat.p", "wb")
        pickle.dump(test_pos_feat, fileObject)
        fileObject.close()

    #POS pattern features
    try:
        fileObject = open(path + "features/feat/train_pos_pattern_feat.p", "r")
        train_pos_pattern_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_pos_pattern_feat = createPOSPatternFeat(X_train)
        fileObject = open(path + "features/feat/train_pos_pattern_feat.p", "wb")
        pickle.dump(train_pos_pattern_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(path + "features/feat/test_pos_pattern_feat.p", "r")
        test_pos_pattern_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_pos_pattern_feat = createPOSPatternFeat(X_test)
        fileObject = open(path + "features/feat/test_pos_pattern_feat.p", "wb")
        pickle.dump(test_pos_pattern_feat, fileObject)
        fileObject.close()

    #unigram features
    try:
        fileObject = open(path + "features/feat/train_unigram_feat.p", "r")
        train_unigram_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_unigram_feat = createUnigramFeat(X_train)
        fileObject = open(path + "features/feat/train_unigram_feat.p", "wb")
        pickle.dump(train_unigram_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(path + "features/feat/test_unigram_feat.p", "r")
        test_unigram_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_unigram_feat = createUnigramFeat(X_test)
        fileObject = open(path + "features/feat/test_unigram_feat.p", "wb")
        pickle.dump(test_unigram_feat, fileObject)
        fileObject.close()

    #bigram features
    try:
        fileObject = open(path + "features/feat/train_bigram_feat.p", "r")
        train_bigram_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_bigram_feat = createBigramFeat(X_train)
        fileObject = open(path + "features/feat/train_bigram_feat.p", "wb")
        pickle.dump(train_bigram_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(path + "features/feat/test_bigram_feat.p", "r")
        test_bigram_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_bigram_feat = createBigramFeat(X_test)
        fileObject = open(path + "features/feat/test_bigram_feat.p", "wb")
        pickle.dump(test_bigram_feat, fileObject)
        fileObject.close()

    #aslog features type_1
    try:
        fileObject = open(path + "features/feat/train_aslog_feat.p", "r")
        train_aslog_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_aslog_feat = createASlogFeat(X_train)
        fileObject = open(path + "features/feat/train_aslog_feat.p", "wb")
        pickle.dump(train_aslog_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(path + "features/feat/test_aslog_feat.p", "r")
        test_aslog_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_aslog_feat = createASlogFeat(X_test)
        fileObject = open(path + "features/feat/test_aslog_feat.p", "wb")
        pickle.dump(test_aslog_feat, fileObject)
        fileObject.close()

    #aslog features type_2
    try:
        fileObject = open(path + "features/feat/train_aslog_2_feat.p", "r")
        train_aslog_2_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_aslog_2_feat = createASlogFeat_2(X_train)
        fileObject = open(path + "features/feat/train_aslog_2_feat.p", "wb")
        pickle.dump(train_aslog_2_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(path + "features/feat/test_aslog_2_feat.p", "r")
        test_aslog_2_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_aslog_2_feat = createASlogFeat_2(X_test)
        fileObject = open(path + "features/feat/test_aslog_2_feat.p", "wb")
        pickle.dump(test_aslog_2_feat, fileObject)
        fileObject.close()

    all_train_feat = np.hstack(
        (train_unigram_feat, train_bigram_feat, train_aslog_feat, train_pos_feat, train_pos_pattern_feat, train_indword_feat))
    all_test_feat = np.hstack(
        (test_unigram_feat, test_bigram_feat, test_aslog_feat, test_pos_feat, test_pos_pattern_feat, test_indword_feat))
    run_classifier(all_train_feat, y_train, all_test_feat,
    path + set_path + '/results/test.txt')

    get_accuracy(X_test, y_test)

def get_accuracy(X_test, y_test):
    y_pred = []
    f = open(path + set_path + '/results/test.txt', 'r')
    for line in f:
        y_pred.append(int(line.strip()))
    true_pred, false_pred = [], []
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            true_pred.append(X_test[i])
        else:
            false_pred.append(X_test[i])

    accuracy = 1.0*len(true_pred)/len(y_pred)
    f = open(path + set_path + '/results/accuracy.txt', 'w+')
    f.write('Accuracy: ' + str(accuracy))
    f = open(path + set_path + '/results/true_predictions.txt', 'w+')
    for i in range(len(true_pred)):
        f.write(X_test[i] + '\n')
        f.write('True: ' + str(y_test[i]) + ', Predicted: ' + str(y_pred[i]) + '\n\n')
    f = open(path + set_path + '/results/false_predictions.txt', 'w+')
    for i in range(len(true_pred)):
        f.write(X_test[i] + '\n')
        f.write('True: ' + str(y_test[i]) + ', Predicted: ' + str(y_pred[i]) + '\n\n')


def run_classifier(X_train, y_train, X_test, predicted_labels_file):
    svc = svm.SVC(decision_function_shape = 'ovo')
    svc.fit(X_train, y_train)
    y_test = svc.predict(X_test)
    with open(predicted_labels_file, "w") as f_out:
        for label in y_test.tolist():
            f_out.write(str(label) + "\n")

def get_labelled_sets(sent_dict):
    sents, labels = [], []
    for key in sent_dict:
        for item in sent_dict[key]:
            if key == 'Other':
                labels.append(0)
            elif key == 'P':
                labels.append(1)
            elif key == 'I':
                labels.append(2)
            elif key == 'O':
                labels.append(3)
            sents.append(item)


    '''for i in range(len(sents)):
        print sents[i]
        print labels[i]'''
    return sents, labels
        
def get_data_sets():
    '''f = open(path + '/data/train.json', 'r')
    for line in f:
        train = json.loads(line)
    f = open(path + '/data/test.json', 'r')
    for line in f:
        test = json.loads(line)
    f = open(path + '/data/dev.json', 'r')
    for line in f:
        dev = json.loads(line)'''
    train, test, dev = {'P': [], 'I': [], 'O': [], 'Other':[]}, {'P': [], 'I': [], 'O': [], 'Other':[]}, {'P': [], 'I': [], 'O': [], 'Other':[]} 
    f = open(path + '/data/all.json', 'r')
    for line in f:
        all = json.loads(line)
    num_samples = len(all["P"])
    train_size, test_size = int(0.6*num_samples), int(0.2*num_samples)

    for key in all:
        train[key] = all[key][:train_size]
    for key in all:
        test[key] = all[key][train_size:(train_size+test_size)]
    for key in all:
        dev[key] = all[key][(train_size+test_size):]

    return train, test, dev

def create_data_sets():
    f = open(path + '/data/int_sents.txt', 'r')
    int_sents = f.readlines()
    f = open(path + '/data/part_sents.txt', 'r')
    part_sents = f.readlines()
    f = open(path + '/data/out_sents.txt', 'r')
    out_sents = f.readlines()

    other_sents = []
    f = open(path + '/data/other_sents.txt', 'r')
    for line in f:
        if line.strip() == '': continue
        other_sents.append(line)

    num_samples = 50000
    train_size, test_size, dev_size = 0.6*num_samples, 0.2*num_samples, 0.2*num_samples

    train, test, dev = {'P': [], 'I': [], 'O': [], 'Other':[]}, {'P': [], 'I': [], 'O': [], 'Other':[]}, {'P': [], 'I': [], 'O': [], 'Other':[]} 
    for i in range(num_samples):
        if i < train_size/4:
            train['P'].append(part_sents[i])
            train['I'].append(int_sents[i])
            train['O'].append(out_sents[i])
            train['Other'].append(other_sents[i])
        elif i < train_size/4 + test_size/4:
            test['P'].append(part_sents[i])
            test['I'].append(int_sents[i])
            test['O'].append(out_sents[i])
            test['Other'].append(other_sents[i])
        elif i < train_size/4 + test_size/4 + dev_size/4:
            dev['P'].append(part_sents[i])
            dev['I'].append(int_sents[i])
            dev['O'].append(out_sents[i])
            dev['Other'].append(other_sents[i])


    f = open(path + '/data/train.json', 'w+')
    f.write(json.dumps(train))
    f = open(path + '/data/test.json', 'w+')
    f.write(json.dumps(test))
    f = open(path + '/data/dev.json', 'w+')
    f.write(json.dumps(dev))

def test():
    part_lines, int_lines, out_lines = [], [], []
    f = open(path + 'features/outcome_caseframes.tsv', 'r')
    count = 0
    for line in f:
        items = line.strip().split('\t')
        freq, pattern = int(items[0]), items[4]
        if freq < 100 or '&&' in pattern: continue
        pattern = pattern.split(':')[-1]
        pattern = pattern.split('__')[-1]
        items = pattern.split('_')
        if len(items) == 1: continue
        pattern = [item.lower() for item in items]
        print ' '.join(item for item in pattern)
        part_lines.append(' '.join(item for item in pattern))
        count += 1
    print count
    f = open(path + 'features/outcome_frames.tsv', 'w+')
    for val in part_lines:
        f.write(val + '\n')

def create_data_from_amt():
    f = open(path + 'data/annotations/AMT/training_mv.json', 'r')
    for line in f:
        dict = json.loads(line)

    sent_dict = {'P': [], 'I': [], 'O': [], 'Other': []}
    count = 0
    for docid in dict:
        count += 1
        if count == 1500: break
        sent_tuples = dict[docid]
        for sent_tuple in sent_tuples:
            sent = [item[0] for item in sent_tuple]
            #print sent
            part_mask = [item[3] for item in sent_tuple]
            int_mask = [item[4] for item in sent_tuple]
            out_mask = [item[5] for item in sent_tuple]


            part_zero, int_zero, out_zero = False, False, False
            if sum(part_mask) == 0: part_zero = True
            if sum(int_mask) == 0: int_zero = True
            if sum(out_mask) == 0: out_zero = True

            '''print part_mask
            print sum(part_mask)
            print sum(int_mask)
            print sum(out_mask)'''
            if (part_zero is True and int_zero is True and out_zero is True):
                sent_dict['Other'].append(' '.join(item for item in sent))
            elif (part_zero is True and int_zero is True and out_zero is False):
                sent_dict['O'].append(' '.join(item for item in sent))
            elif (part_zero is True and int_zero is False and out_zero is True):
                sent_dict['I'].append(' '.join(item for item in sent))
            elif (part_zero is False and int_zero is True and out_zero is True):
                sent_dict['P'].append(' '.join(item for item in sent))

    print len(sent_dict['Other'])
    print len(sent_dict['P'])
    print len(sent_dict['I'])
    print len(sent_dict['O'])

    f = open(path + 'data/all.json', 'w+')
    f.write(json.dumps(sent_dict))
    
if __name__ == '__main__':
    train, test, dev = get_data_sets()
    X_train, y_train = get_labelled_sets(train)
    X_test, y_test = get_labelled_sets(test)
    prepare_features(X_train, y_train, X_test, y_test)





