import sys
import logging
import json
import numpy as np
from collections import defaultdict, namedtuple, Counter
import re
from sklearn import svm
from sklearn.svm import LinearSVC
import pickle

path = '/nlp/data/romap/law/task_3/models/shell/'

def createUnigramFeat(words, word_vecs):
    all_feat = []
    for word in words:
        all_feat.append(word_vec(word))
    return all_feat

def createCharFeat(words):
    all_feat = []; chars = [c for c in string.ascii_lowercase]
    for word in words:
        vec = [0 for item in chars]
        for i in range(len(chars)):
            for char in word: if char == chars[i]: vec[i] += 1
            
        total = np.sum(vec)
        for i in range(len(vec)):
            vec[i] = (1.0*vec[i])/total
        all_feat.append(vec)
    return all_feat

ef prepare_features(X_train, y_train, X_test, y_test, word_vecs):
        
    #unigram features       
    try:
        fileObject = open(dataset_path + "/features/feat/train_unigram_feat.p", "r")
        train_unigram_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_unigram_feat = createUnigramFeat(X_train, word_vecs)
        fileObject = open(dataset_path + "/features/feat/train_unigram_feat.p", "wb")
        pickle.dump(train_unigram_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(dataset_path + "/features/feat/test_unigram_feat.p", "r")
        test_unigram_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_unigram_feat = createUnigramFeat(X_test, word_vecs)
        fileObject = open(dataset_path + "/features/feat/test_unigram_feat.p", "wb")
        pickle.dump(test_unigram_feat, fileObject)
        fileObject.close()

    print 'Unigram done!'
    print 'Stacking features!'
    rows = len(y_train)
    all_train_feat = np.reshape(train_unigram_feat, (rows, -1));
    rows = len(y_test)
    all_test_feat = np.reshape(test_unigram_feat, (rows, -1))

    print 'Running classifier!'
    run_classifier(all_train_feat, y_train, all_test_feat, y_test,
    path + '/results/test.txt')
    print 'Done!'

def run_classifier(X_train, y_train, X_test, y_test, predicted_labels_file):
    svc = svm.SVC(decision_function_shape = 'ovo')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    confusion = []
    with open(predicted_labels_file, "w") as f_out:
        for i in range(len(y_pred.tolist())):
            label, true_label = y_pred.tolist()[i], y_test[i]
            f_out.write(str(label) + ' ' + str(true_label) + "\n")

def get_embeddings(vocab, wvec_path):
    word_vecs = {}
    if wvec_path[-3:] == 'bin':
        with open(wvec_path, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)

    elif wvec_path[-3:] == 'txt':
        f = open(wvec_path, 'r'); lines = f.readlines()[1:]
        for line in lines:
            items = line.strip().split(' ')
            word, vec = items[0], [float(item) for item in items[1:]]
            if word in vocab: word_vecs[word] = vec
                
    return word_vecs

if __name__ == '__main__':
    global set_name, wvec_path
    wvec_paths = {'google': '/nlp/data/corpora/GoogleNews-vectors-negative300.bin',
                                   'legal': '/nlp/data/romap/ambig/w2v/w2v100-300.txt',
                                   'concept': '/nlp/data/romap/conceptnet/numberbatch-en-17.06.txt'
                                   }
    set_name = sys.argv[1]; wvec_path = wvec_paths[set_name]
