from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import logging
import json
import numpy as np
import sys
import os

path = '/nlp/data/romap/law/task_3/models_3/shell/'
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
            word, vec = items[0], np.array([float(item) for item in items[1:]], dtype='float32')
            if word in vocab: word_vecs[word] = vec
    return word_vecs

def get_datasets():
    f = open('/nlp/data/romap/law/word_labels_' + set_name + '_3.tsv', 'r')
    data, tokens, tags = [], [], []
    prev, cur = 1, 1
    lines = f.readlines()
    for line in lines:
        items = line.strip().split('\t')
        cur = int(items[4])
        if cur > prev:
            data.append((tokens, tags))
            tokens, tags, prev = [], [], cur
        else: tokens.append(items[0]); tags.append(items[1])

    # specificity news data
    '''
    f = open('/nlp/data/romap/law/task_3/data/spec.json', 'r')
    for line in f: temp = json.loads(line)

    data = [(temp[key]['tokens'], temp[key]['tags']) for key in temp]
    tag_idx = {'I': 0, 'O': 1}
    cutoff = len(data) - int(0.1*len(data))
    train_data, test_data = data[:cutoff], data[cutoff:]
    X_train, y_train, X_test, y_test = [], [], [], []
    
    for item in train_data:
        X_train.extend([word.lower() for word in item[0]])
        y_train.extend([tag_idx[tag] for tag in item[1]])

    for item in test_data:
        X_test.extend([word.lower() for word in item[0]])
        y_test.extend([tag_idx[tag] for tag in item[1]])
    
    return X_train, y_train, X_test, y_test
    '''

    return data

def prepare_data(fold, data, word_vecs):

    #segreagte by titles
    '''
    title_split, titles = [str(i) for i in range(1, 55)], []
    bin_val = len(titles)/10 + 1
    title_folds = [titles[i: i+bin_val] for i in range(0, len(titles), bin_val)]
    
    f = open('/nlp/data/romap/law/word_labels_' + set_name + '_3.tsv', 'r')
    lines = f.readlines(); t = []
    for line in lines:
        items = line.strip().split('\t')
        if len(items) < 2:
            t = []
            titles.append(t)
        else: t.append(items[2]); 

    test_titles = []
    '''        
    
    bin_val = len(data)/100 + 1
    folds = [data[i: i+bin_val] for i in range(0, len(data), bin_val)]
    print 'binval: ' + str(bin_val)
    print 'folds: ' + str(len(folds))
    
    testing_data = folds[fold]; training_data = [folds[i] for i in range(0, 10) if i != fold]
    training_data = [item for sublist in training_data for item in sublist]



    print 'testing_data'
    print testing_data
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

    return X_train, y_train, X_test, y_test, train_words, test_words

def train(X_train, y_train, X_test, y_test, fold, word_vecs, test_words):

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)


    # test on specificity
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
    # change this back for law data

    y = log_reg.predict(X_test)
    #y = [1 if item > 0 else 0 for item in y]
    
    temp = {'0': {'tokens': test_words, 'pred': list(y), 'true': list(y_test)}}

    dirpath = path + set_name + '/' + 'results/' + model_name + '/'
    if os.path.isdir(dirpath) is False: os.mkdir(dirpath)
    f = open(dirpath + 'results-' + str(fold) + '.json', 'w+')
    f.write(json.dumps(temp)); f.close()


if __name__ == '__main__':
    global set_name, wvec_path, model_name



    wvec_paths = {'google': '/nlp/data/corpora/GoogleNews-vectors-negative300.bin',
                                   'legal': '/nlp/data/romap/ambig/w2v/w2v100-300.txt',
                                   'concept': '/nlp/data/romap/conceptnet/numberbatch-en-17.06.txt'
                                   }
    set_name = sys.argv[1]; wvec_path = wvec_paths[set_name]; model_name = sys.argv[2]
    print set_name

    #X_train, y_train, X_test, y_test = get_datasets()
    data = get_datasets()
    X = [item[0] for item in data]; y = [item[1] for item in data]
    X = [item for sublist in X for item in sublist]; y = [item for sublist in y for item in sublist]    
    vocab = list(set(X)); 
    word_vecs = get_embeddings(vocab, wvec_path)


    for fold in range(0, 10):
        print fold
        X_train, y_train, X_test, y_test, train_words, test_words = prepare_data(fold, data, word_vecs)
        train(X_train, y_train, X_test, y_test, fold, word_vecs, test_words)
