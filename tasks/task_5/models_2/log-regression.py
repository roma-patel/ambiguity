from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import logging
import json
import numpy as np
import sys
import os
import pickle

path = '/nlp/data/romap/law/task_5/models_2/shell/'
results_path = '/nlp/data/romap/law/task_5/models_2/title_results/'
tm_path = '/nlp/data/romap/law/task_5/models_2/title_models/'
bigram_results_path = '/nlp/data/romap/law/task_5/models_2/title_bigrams/'

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

def get_bigram_wvec():
    word_vecs = {}; concept = True
    '''
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
    '''
    bigrams = []
    f = open('/nlp/data/romap/law/fin_bigrams.txt', 'r')
    lines = f.readlines(); bigrams = [line.strip() for line in lines]
    
    if concept is True:
        f = open(wvec_path, 'r'); lines = f.readlines()[1:]
        for line in lines:
            items = line.strip().split(' ')
            word, vec = items[0], np.array([float(item) for item in items[1:]], dtype='float32')
            if word in bigrams: word_vecs[word] = vec
    return word_vecs

def get_datasets():
    train_path = '/nlp/data/romap/law/task_4/data/titles/' + title_name + '/train.txt'
    test_path = '/nlp/data/romap/law/task_4/data/titles/all_test.txt'
    
    f = open(train_path, 'r')
    data, tokens, tags = [], [], []
    prev, cur = 1, 1
    lines = f.readlines()
    for line in lines:
        items = line.strip().split('\t')
        cur = int(items[4])
        if cur == 1: prev = 1

        if cur > prev:
            data.append((tokens, tags))
            tokens, tags, prev = [], [], cur
        else: tokens.append(items[0]); tags.append(items[1])


    data.append((['united', 'states'], ['D', 'D']))

    return data

def prepare_data(fold, data, word_vecs):

    training_data, testing_data = data, []
    test_path = '/nlp/data/romap/law/task_4/data/titles/all_test.txt'
    
    f = open(test_path, 'r')
    tokens, tags = [], []; prev, cur = 1, 1
    lines = f.readlines()
    for line in lines:
        items = line.strip().split('\t')
        cur = int(items[4])
        if cur == 1: prev = 1

        if cur > prev:
            testing_data.append((tokens, tags))
            tokens, tags, prev = [], [], cur
        else: tokens.append(items[0]); tags.append(items[1])
    

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
    tag_to_idx = {'D': 0, 'G': 0, 'O': 1}

    y_train = [tag_to_idx[tag] for tag in y_train]
    y_test = [tag_to_idx[tag] for tag in y_test]

    return X_train, y_train, X_test, y_test, train_words, test_words

def train(X_train, y_train, X_test, y_test, fold, word_vecs, test_words):

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    #save trained models
    if os.path.isdir(tm_path) is False: os.mkdir(tm_path)
    tm_name = tm_path + title_name + '_model.sav'
    pickle.dump(log_reg, open(tm_name, 'wb'))
    
    y = log_reg.predict(X_test)
    #y = [1 if item > 0 else 0 for item in y]
    
    temp = {'0': {'tokens': test_words, 'pred': list(y), 'true': list(y_test)}}

    dirpath = results_path + title_name + '/'
    if os.path.isdir(dirpath) is False: os.mkdir(dirpath)

    dirpath += set_name + '/' 
    if os.path.isdir(dirpath) is False: os.mkdir(dirpath)
    dirpath += model_name + '/'
    if os.path.isdir(dirpath) is False: os.mkdir(dirpath)

    f = open(dirpath + 'results-' + str(fold) + '.json', 'w+')
    f.write(json.dumps(temp)); f.close()

def train_bigram(X_train, y_train, X_test, y_test, fold, word_vecs, test_words):
    filename = tm_path + title_name + '_model.sav'

    X_test, y_test = [], []

    f = open('/nlp/data/romap/law/fin_bigrams.txt', 'r')
    lines = f.readlines(); test_words = [line.strip() for line in lines]
    fin_words = []
    for word in test_words:
        if word not in word_vecs.keys(): continue
        fin_words.append(word)
        y_test.append(0)
        X_test.append(word_vecs[word])
    for i in range(len(X_test)): X_test[i] = np.array(X_test[i])

    log_reg = pickle.load(open(filename, 'rb'))
    y = log_reg.predict(X_test)
    temp = {'0': {'tokens': fin_words, 'pred': list(y), 'true': list(y_test)}}

    dirpath = bigram_results_path + title_name + '/'
    if os.path.isdir(dirpath) is False: os.mkdir(dirpath)

    dirpath += set_name + '/' 
    if os.path.isdir(dirpath) is False: os.mkdir(dirpath)
    dirpath += model_name + '/'
    if os.path.isdir(dirpath) is False: os.mkdir(dirpath)
    f = open(dirpath + 'results-' + str(fold) + '.json', 'w+')
    f.write(json.dumps(temp)); f.close()

if __name__ == '__main__':
    
    global set_name, wvec_path, model_name, title_name
    # add title arg!


    wvec_paths = {'google': '/nlp/data/corpora/GoogleNews-vectors-negative300.bin',
                                   'legal': '/nlp/data/romap/ambig/w2v/w2v100-300.txt',
                                   'concept': '/nlp/data/romap/conceptnet/numberbatch-en-17.06.txt'
                                   }
    set_name = sys.argv[1]; wvec_path = wvec_paths[set_name]; model_name = sys.argv[2]
    title_name = sys.argv[3]


    #X_train, y_train, X_test, y_test = get_datasets()
    data = get_datasets()
    X = [item[0] for item in data]; y = [item[1] for item in data]
    X = [item for sublist in X for item in sublist]; y = [item for sublist in y for item in sublist]    
    vocab = list(set(X)); 
    word_vecs = get_embeddings(vocab, wvec_path)


    fold = 0
    X_train, y_train, X_test, y_test, train_words, test_words = prepare_data(fold, data, word_vecs)
    train(X_train, y_train, X_test, y_test, fold, word_vecs, test_words)
    word_vecs = get_bigram_wvec()
    train_bigram(X_train, y_train, X_test, y_test, fold, word_vecs, test_words)
      
    '''
    for i in range(1, 55):
        f = open('/Users/romapatel/Desktop/law/task_5/shell_scripts/models_2/run-log-' + str(i) + '.sh', 'w+')
        f.write('source /home1/r/romap/crf/crf_task/bin/activate\n')
        f.write('python /nlp/data/romap/law/task_5/models_2/log-regression.py concept log ' + str(i) + '\n')
     '''   



        
