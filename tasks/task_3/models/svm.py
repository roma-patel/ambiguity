from collections import defaultdict, namedtuple, Counter
import sys
import logging
import json
import numpy as np
import re
from sklearn import svm
from sklearn.svm import LinearSVC
import pickle
import os

path = '/nlp/data/romap/law/task_3/models/shell/'

def createUnigramFeat(word_vecs):
    all_feat = []
    for word_vec in word_vecs:
        all_feat.append(word_vec)
    return all_feat

'''
def createCharFeat(words):
    all_feat = []; chars = [c for c in string.ascii_lowercase]
    for word in words:
        vec = [0 for item in chars]
        for i in range(len(chars)):
            for char in word:
                if char == chars[i]: vec[i] += 1
            
        total = np.sum(vec)
        for i in range(len(vec)):
            vec[i] = (1.0*vec[i])/total
        all_feat.append(vec)
    return all_feat
'''

def prepare_features(X_train, y_train, X_test, y_test, fold, test_words):
    feat_path = path + set_name + '/features/feat/' + str(fold) + '/'
    if os.path.isdir(feat_path) is False: os.mkdir(feat_path)
    #unigram features       
    try:
        fileObject = open(feat_path + "/train_unigram_feat.p", "r")
        train_unigram_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        train_unigram_feat = createUnigramFeat(X_train)
        fileObject = open(feat_path + "/train_unigram_feat.p", "wb")
        pickle.dump(train_unigram_feat, fileObject)
        fileObject.close()
    try:
        fileObject = open(feat_path + "/test_unigram_feat.p", "r")
        test_unigram_feat = pickle.load(fileObject)
        fileObject.close()

    except IOError as e:
        test_unigram_feat = createUnigramFeat(X_test)
        fileObject = open(feat_path + "/test_unigram_feat.p", "wb")
        pickle.dump(test_unigram_feat, fileObject)
        fileObject.close()

    print 'Unigram done!'
    print 'Stacking features!'
    rows = len(y_train)
    all_train_feat = np.reshape(train_unigram_feat, (rows, -1));
    rows = len(y_test)
    all_test_feat = np.reshape(test_unigram_feat, (rows, -1))

    print 'Running classifier!'
    dirpath = path + set_name + '/' + 'results/' + model_name + '/'
    if os.path.isdir(dirpath) is False: os.mkdir(dirpath)
    
    run_classifier(test_words, all_train_feat, y_train, all_test_feat, y_test,
    dirpath + '/results-' + str(fold) + '.txt')
    print 'Done!'

def run_classifier(test_words, X_train, y_train, X_test, y_test, predicted_labels_file):
    svc = svm.SVC(decision_function_shape = 'ovo')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    confusion = []
    with open(predicted_labels_file, "w") as f_out:
        for i in range(len(y_pred.tolist())):
            label, true_label = y_pred.tolist()[i], y_test[i]
            f_out.write(str(label) + ' ' + str(true_label) + ' ' + test_words[i] +"\n")

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

def get_datasets():
    f = open('/nlp/data/romap/law/word_labels_' + set_name + '.tsv', 'r')
    data, tokens, tags = [], [], []
    lines = f.readlines()
    for line in lines[2:]:
        items = line.strip().split('\t')
        if len(items) < 2:
            data.append((tokens, tags))
            tokens, tags = [], []
        else: tokens.append(items[0]); tags.append(items[1])
    return data

    

def prepare_data(fold, data, word_vecs):
    bin_val = len(data)/10 + 1
    folds = [data[i: i+bin_val] for i in range(0, len(data), bin_val)]
        
    testing_data = folds[fold]; training_data = [folds[i] for i in range(0, 10) if i != fold]
    training_data = [item for sublist in training_data for item in sublist]

    X_train = [item[0] for item in training_data]; y_train = [item[1] for item in training_data]
    X_test = [item[0] for item in testing_data]; y_test = [item[1] for item in testing_data]
    X_train = [item for sublist in X_train for item in sublist]; y_train = [item for sublist in y_train for item in sublist]
    X_test = [item for sublist in X_test for item in sublist]; y_test = [item for sublist in y_test for item in sublist]

    train_words = X_train; test_words = [item for item in X_test]
    for i in range(len(X_train)):
        if X_train[i] in word_vecs.keys():
            X_train[i] = word_vecs[X_train[i]]
        else:
            X_train[i] = np.array([np.random.rand(1)[0] for j in range(0, 300)], dtype='float32')


    for i in range(len(X_test)):
        if X_test[i] in word_vecs.keys():
            X_test[i] = word_vecs[X_test[i]]
        else:
            X_test[i] = np.array([np.random.rand(1)[0] for j in range(0, 300)], dtype='float32')

    tag_to_idx = {'D': 0, 'G': 1, 'O': 2}
    y_train = [tag_to_idx[tag] for tag in y_train]
    y_test = [tag_to_idx[tag] for tag in y_test]


    # specificity data
    '''
    f = open('/nlp/data/romap/law/task_3/data/spec.json', 'r')
    for line in f: temp = json.loads(line)
    data = [(temp[key]['tokens'], temp[key]['tags']) for key in temp]
    X_test, y_test, tag_idx = [], [], {'I': 0, 'O': 1}    
    for item in data:
        X_test.extend([word.lower() for word in item[0]])
        y_test.extend([tag_idx[tag] for tag in item[1]])

    for i in range(len(X_test)):
        if X_test[i] in word_vecs.keys():
            X_test[i] = word_vecs[X_test[i]]
        else:
            X_test[i] = np.array([np.random.rand(1)[0] for j in range(0, 300)], dtype='float32')
    '''
    # change back for law data
    
    return X_train, y_train, X_test, y_test, train_words, test_words



if __name__ == '__main__':
    global set_name, wvec_path, model_name
    wvec_paths = {'google': '/nlp/data/corpora/GoogleNews-vectors-negative300.bin',
                                   'legal': '/nlp/data/romap/ambig/w2v/w2v100-300.txt',
                                   'concept': '/nlp/data/romap/conceptnet/numberbatch-en-17.06.txt'
                                   }
    set_name = sys.argv[1]; wvec_path = wvec_paths[set_name]; model_name = sys.argv[2]
    data = get_datasets()
    X = [item[0] for item in data]; y = [item[1] for item in data]
    X = [item for sublist in X for item in sublist]; y = [item for sublist in y for item in sublist]    
    vocab = list(set(X)); 
    word_vecs = get_embeddings(vocab, wvec_path)


    for fold in range(0, 10):
        print fold
        X_train, y_train, X_test, y_test, train_words, test_words = prepare_data(fold, data, word_vecs)
        prepare_features(X_train, y_train, X_test, y_test, fold, test_words)
