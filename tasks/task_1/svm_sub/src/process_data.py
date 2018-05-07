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
#from sklearn.cross_validation import KFold, cross_val_score
#from sklearn.cross_validation import StratifiedKFold

path = '/nlp/data/romap/law/task_1/'

def create_sent_dict():
    f = open('/nlp/data/romap/law/task_1/data/text_dict.json', 'r')
    for line in f: text_dict = json.loads(line)

    '''f = open('/Users/romapatel/Desktop/supreme_court/2009/BURLINGTON NORTHERN & SANTA FE RAILWAY CO. ET AL. v. UNITED STATES ET AL..txt', 'r')
    lines = f.readlines()
    text_dict = {'supreme': {'2017': {'case': {'lines': lines}}}}'''

    words_dict = {'supreme': {}, 'circuit': {}}
    #supreme
    for year in text_dict['supreme']:
        words_dict['supreme'][year] = {}
        for case in text_dict['supreme'][year]:
            lines = text_dict['supreme'][year][case]['lines']
            words_dict['supreme'][year][case] = {'parties': '', 'sents': []}
            
            words, start_indices, end_indices = [], [], []
            for i in range(len(lines)):
                line = lines[i]
                if 'writ of certiorari' in line or 'writs of certiorari' in line: start_indices.append(i)
                if 'FOOTNOTE' in line or 'ootnote' in line and i > 100: end_indices.append(i)

            if len(start_indices) < 1: start = 30
            else: start = start_indices[0]+1
            if len(end_indices) < 1: end = len(lines)
            else: end = end_indices[0]

            for line in lines[start:end]:
                line = line.strip(); words = line.split(' ')
                if len(words) <= 10: continue
                words_dict['supreme'][year][case]['sents'].append(line)

    #circuit
    for court in text_dict['circuit']:
        words_dict['circuit'][court] = {}
        for year in text_dict['circuit'][court]:
            words_dict['circuit'][court][year] = {}
            for case in text_dict['circuit'][court][year]:
                words_dict['circuit'][court][year][case] = {'parties': '', 'sents': []}
                lines = text_dict['circuit'][court][year][case]['lines']
                for line in lines:
                    line = line.strip(); words = line.split(' ')
                    if len(words) <= 10: continue
                    words_dict['circuit'][court][year][case]['sents'].append(line)
                    
    f = open('/nlp/data/romap/law/task_1/data/sents_dict.json', 'w+')
    f.write(json.dumps(words_dict))    

                
def create_word_dict():
##    ADD STOPWORDS!
    stopwords = []
    f = open('/nlp/data/romap/law/task_1/data/stopwords.txt', 'r')
    for line in f: stopwords.append(line.strip())
    
    f = open('/nlp/data/romap/law/task_1/data/sents_dict.json', 'r')
    for line in f: text_dict = json.loads(line)

    words_dict = {'supreme': {}, 'circuit': {}}
    #supreme
    for year in text_dict['supreme']:
        words_dict['supreme'][year] = {}
        for case in text_dict['supreme'][year]:
            lines = text_dict['supreme'][year][case]['sents']
            words_dict['supreme'][year][case] = {'words': []}
            words = []
            for line in lines:
                items = line.lower().split(' ')
                #remove non-ascii
                for item in items:
                    item = re.sub('[,.:;?\"\[\]\(\)]', '', item)
                    if ' ' in item:
                        new_items = item.split(' ')
                        items.extend(new_items)
                        continue                       
                    if '\\' in item:
                        new_items = item.split('\\')
                        items.extend(new_items)
                        continue
                    if '--' in item:
                        new_items = item.split('--')
                        items.extend(new_items)
                        continue
                    if '\'s' in item: item = item.split('\'s')[0]
                    if re.search('[0-9]', item) is not None: continue
                    if len(item) < 3: continue
                    if re.search('[a-z]', item[0]) is None: item = item[1:]
                    if re.search('[a-z]', item[-1]) is None: item = item[:-1]
                    if len(item) < 3: continue
                    if item in stopwords: continue
                    words.append(item)
                #words.extend(line.lower().split(' '))
            freq = FreqDist(word for word in words)
            words_dict['supreme'][year][case]['words'] = freq

    #circuit
    for court in text_dict['circuit']:
        words_dict['circuit'][court] = {}
        for year in text_dict['circuit'][court]:
            words_dict['circuit'][court][year] = {}
            for case in text_dict['circuit'][court][year]:
                words_dict['circuit'][court][year][case] = {'words': []}
                lines = text_dict['circuit'][court][year][case]['sents']
                words = []
                for line in lines:
                    items = line.lower().split(' ')
                    #remove non-ascii
                    for item in items:
                        item = re.sub('[,.\:;?\"\[\]\(\)]', '', item)

                        if ' ' in item:
                            new_items = item.split(' ')
                            items.extend(new_items)
                            continue                       
                        if '\\' in item:
                            new_items = item.split('\\')
                            items.extend(new_items)
                            continue
                        if '--' in item:
                            new_items = item.split('--')
                            items.extend(new_items)
                            continue
                        if '\'s' in item: item = item.split('\'s')[0]                    
                        if re.search('[0-9]', item) is not None: continue
                        if len(item) < 3: continue

                        if re.search('[a-z]', item[0]) is None: item = item[1:]
                        if re.search('[a-z]', item[-1]) is None: item = item[:-1]
                        if len(item) < 3: continue
                        if item in stopwords: continue
                        words.append(item)
                    #words.extend(line.lower().split(' '))
                freq = FreqDist(word for word in words)
                words_dict['circuit'][court][year][case]['words'] = freq

    f = open('/nlp/data/romap/law/task_1/data/words_dict_2.json', 'w+')
    f.write(json.dumps(words_dict))

def get_top_words():    
    f = open('/nlp/data/romap/law/task_1/data/words_dict_2.json', 'r')
    for line in f: text_dict = json.loads(line)

    supreme_words, circuit_words, all_words = {}, {}, {}
    #supreme
    for year in text_dict['supreme']:
        for case in text_dict['supreme'][year]:
            words = text_dict['supreme'][year][case]['words']
            for word in words:
                count = text_dict['supreme'][year][case]['words'][word]
                if word not in supreme_words.keys(): supreme_words[word] = 0
                if word not in all_words.keys(): all_words[word] = 0
                all_words[word] += count
                supreme_words[word] += count

    #circuit
    for court in text_dict['circuit']:
        for year in text_dict['circuit'][court]:
            for case in text_dict['circuit'][court][year]:
                words = text_dict['circuit'][court][year][case]['words']
                for word in words:
                    count = text_dict['circuit'][court][year][case]['words'][word]
                    if word not in circuit_words.keys(): circuit_words[word] = 0
                    if word not in all_words.keys(): all_words[word] = 0
                    all_words[word] += count
                    circuit_words[word] += count
    fin_dict = {'supreme':supreme_words, 'circuit': circuit_words}
    f = open('/nlp/data/romap/law/task_1/data/top_words_2.json', 'w+')
    f.write(json.dumps(fin_dict))

    f = open('/nlp/data/romap/law/task_1/data/all_top_words_2.json', 'w+')
    f.write(json.dumps(all_words))
                   

def copy_data():
    f = open('/nlp/data/romap/law/task_1/data/datasets.json', 'r')
    for line in f: data = json.loads(line)

    f = open('/nlp/data/romap/law/task_1/data/index_dict.json', 'r')
    for line in f: index_dict = json.loads(line)

    index_flipped = {'supreme': {}, 'circuit': {}}
    for court in index_dict['circuit']:
        #index_flipped['circuit'][court] = {}
        for year in index_dict['circuit'][court]:
            #index_flipped['circuit'][court][year] = {}
            for case in index_dict['circuit'][court][year]:
                path = '/nlp/data/romap/law/data/supreme_court/' + year + '/' + case
                val = index_dict['circuit'][court][year][case]
                #index_flipped['circuit'][court][year][val] = case
                if val not in index_flipped['circuit'].keys():
                    index_flipped['circuit'][val] = []
                index_flipped['circuit'][val].append(case)
                    

    for year in index_dict['supreme']:
        #index_flipped['supreme'][year] = {}
        for case in index_dict['supreme'][year]:
            val = index_dict['supreme'][year][case]
            #index_flipped['supreme'][year][val] = case
            if val not in index_flipped['supreme'].keys():
                index_flipped['supreme'][val] = []
            index_flipped['supreme'][val].append(case)
                

    f = open('/nlp/data/romap/law/task_1/data/index_flipped.json', 'w+')
    f.write(json.dumps(index_flipped))
    '''denied, supreme, circuit = [], [], []
    f = open('/nlp/data/romap/law/task_1/data/denied_index.txt', 'r')
    for line in f: denied.append(line.strip())

    #train
    for val in data['train']:
        supreme = data['train']['supreme']
        circuit = data['train']['circuit']'''
        
        
def tf_idf_words():
    f = open('/nlp/data/romap/law/task_1/data/words_dict_2.json', 'r')
    for line in f: fin_dict = json.loads(line)
    num_docs = 0; vocab = []; doc_freq = {}


    
    '''for year in fin_dict['supreme']:
        num_docs += len(fin_dict['supreme'][year].keys())
        for case in fin_dict['supreme'][year]:
            vocab.extend(fin_dict['supreme'][year][case]['words'].keys())

    vocab = list(set(vocab))
    for court in fin_dict['circuit']:
        for year in fin_dict['circuit'][court]:
            num_docs += len(fin_dict['circuit'][court][year].keys())
            for case in fin_dict['circuit'][court][year]:
                vocab.extend(fin_dict['circuit'][court][year][case]['words'].keys())
    vocab = list(set(vocab))

    for word in vocab:
        doc_freq[word] = 0
        for year in fin_dict['supreme']:
            for case in fin_dict['supreme'][year]:
                if word in fin_dict['supreme'][year][case]['words'].keys():
                    doc_freq[word] += 1

        for court in fin_dict['circuit']:
            for year in fin_dict['circuit'][court]:
                for case in fin_dict['circuit'][court][year]:
                    if word in fin_dict['circuit'][court][year][case]['words'].keys():
                        doc_freq[word] += 1

    f = open('/nlp/data/romap/law/task_1/data/files/doc_freq.json', 'w+')
    f.write(json.dumps(doc_freq))
    f = open('/nlp/data/romap/law/task_1/data/files/doc_freq.txt', 'w+')

    sorted_x = sorted(doc_freq.items(), key=operator.itemgetter(1), reverse=True)
    for item in sorted_x:
        f.write(item[0] + '\t' + str(item[1]) + '\n')'''


    
    f = open('/nlp/data/romap/law/task_1/data/files/doc_freq.json', 'r')
    for line in f: doc_freq = json.loads(line)

    #supreme tf-idf
    tfidf = {'supreme': {}, 'circuit': {}}
    for year in fin_dict['supreme']:
        tfidf['supreme'][year] = {}
        for case in fin_dict['supreme'][year]:

            tfidf['supreme'][year][case] = {'words': {}}
            words = fin_dict['supreme'][year][case]['words']
            for word in words:
                tf = words[word]; df = doc_freq[word]
                
                idf = np.log((1.0*num_docs)/df)
                tfidf['supreme'][year][case]['words'][word] = tf*idf
            

    for court in fin_dict['circuit']:
        tfidf['circuit'][court] = {}
        for year in fin_dict['circuit'][court]:
            tfidf['circuit'][court][year] = {}
            for case in fin_dict['circuit'][court][year]:
                tfidf['circuit'][court][year][case] = {'words': {}}
                words = fin_dict['circuit'][court][year][case]['words']
                for word in words:
                    tf = words[word]; df = doc_freq[word]
                    idf = np.log((1.0*num_docs)/df)
                    tfidf['circuit'][court][year][case]['words'][word] = tf*idf

    f = open('/nlp/data/romap/law/task_1/data/files/tfidf.json', 'w+')
    f.write(json.dumps(tfidf))


def top_tfidf(num_words):
    f = open('/nlp/data/romap/law/task_1/data/files/tfidf.json', 'r')
    for line in f: fin_dict = json.loads(line)

    vocab = []
    
    for year in fin_dict['supreme']:
        for case in fin_dict['supreme'][year]:
            words = fin_dict['supreme'][year][case]['words']
            sorted_x = sorted(words.items(), key = operator.itemgetter(1), reverse=True)
            for item in sorted_x[:num_words]:
                vocab.append(item[0])
    vocab = list(set(vocab))

    for court in fin_dict['circuit']:
        for year in fin_dict['circuit'][court]:
            num_docs += len(fin_dict['circuit'][court][year].keys())
            for case in fin_dict['circuit'][court][year]:
                words = fin_dict['circuit'][court][year][case]['words']
                sorted_x = sorted(words.items(), key = operator.itemgetter(1), reverse=True)
            for item in sorted_x[:num_words]:
                vocab.append(item[0])
    vocab = list(set(vocab))

    f = open('/nlp/data/romap/law/task_1/data/files/unigrams_' + str(num_words) + '.txt', 'w+')
    for word in vocab: f.write(word + '\n')

def get_sections(sentences):
    titles, sections, dict = [], [], {}

    current, flag = 'None', True
    for sentence in sentences:
        words = sentence.split(' ')
        for i in range(len(words)):
            word = words[i]
            if 'U.' in word and 'S.' in word and 'C.' in word:
                #supreme
                '''if i-1 < 0 or i+1 >= len(words): continue
                title, sec = words[i-1], words[i+1]
                if title not in dict.keys(): dict[title] = []
                dict[title].append(sec)'''
                #circuit
                if i-1 < 0 or i+2 >= len(words): continue
                title, sec = words[i-1], words[i+2]
                if title not in dict.keys(): dict[title] = []
                dict[title].append(sec)
                
    #return [list(set(titles)), list(set(sections))]
    return dict

def get_metadata():
    f = open('/nlp/data/romap/law/task_1/data/index_dict.json', 'r')
    for line in f: index_dict = json.loads(line)

    '''for year in index_dict['supreme']:
        for case in index_dict['supreme'][year]:
            dict = {'sections': {}}
            index = index_dict['supreme'][year][case]
            if year == '2017':
                new_filepath = '/nlp/data/romap/law/task_1/data/test/docs/positive/' + index + '.txt'
                meta_filepath = '/nlp/data/romap/law/task_1/data/test/meta/positive/' + index + '.json'

            else:
                new_filepath = '/nlp/data/romap/law/task_1/data/train/docs/positive/' + index + '.txt'
                meta_filepath = '/nlp/data/romap/law/task_1/data/train/meta/positive/' + index + '.json'

            newf = open(new_filepath, 'r')
            lines = newf.readlines()
            dict['sections'] = get_sections(lines)
            f = open(meta_filepath, 'w+')
            f.write(json.dumps(dict))'''
            

    for court in index_dict['circuit']:
        for year in index_dict['circuit'][court]:
            for case in index_dict['circuit'][court][year]:
                dict = {'sections': {}}
                index = index_dict['circuit'][court][year][case]
 
                if year == '2017':
                    new_filepath = '/nlp/data/romap/law/task_1/data/test/docs/negative/' + index + '.txt'
                    meta_filepath = '/nlp/data/romap/law/task_1/data/test/meta/negative/' + index + '.json'

                else:
                    new_filepath = '/nlp/data/romap/law/task_1/data/train/docs/negative/' + index + '.txt'
                    meta_filepath = '/nlp/data/romap/law/task_1/data/train/meta/negative/' + index + '.json'

                newf = open(new_filepath, 'r')
                lines = newf.readlines()
                
                dict['sections'] = get_sections(lines)
                f = open(meta_filepath, 'w+')
                f.write(json.dumps(dict))
                

def testing_tf_idf_words(court_num):
    court = court_num + '_circuit'
    f = open('/nlp/data/romap/law/task_1/data/words_dict_2.json', 'r')
    for line in f: fin_dict = json.loads(line)
    num_docs = 0; vocab = []; doc_freq = {}

    for year in fin_dict['circuit'][court]:
        num_docs += len(fin_dict['circuit'][court][year].keys())
        for case in fin_dict['circuit'][court][year]:
            vocab.extend(fin_dict['circuit'][court][year][case]['words'].keys())
    vocab = list(set(vocab))

    for word in vocab:
        doc_freq[word] = 0
        for year in fin_dict['circuit'][court]:
            for case in fin_dict['circuit'][court][year]:
                if word in fin_dict['circuit'][court][year][case]['words'].keys():
                    doc_freq[word] += 1
                    
    '''for year in fin_dict['supreme']:
        num_docs += len(fin_dict['supreme'][year].keys())
        for case in fin_dict['supreme'][year]:
            vocab.extend(fin_dict['supreme'][year][case]['words'].keys())

    vocab = list(set(vocab))
    for court in fin_dict['circuit']:
        for year in fin_dict['circuit'][court]:
            num_docs += len(fin_dict['circuit'][court][year].keys())
            for case in fin_dict['circuit'][court][year]:
                vocab.extend(fin_dict['circuit'][court][year][case]['words'].keys())
    vocab = list(set(vocab))

    for word in vocab:
        doc_freq[word] = 0
        for year in fin_dict['supreme']:
            for case in fin_dict['supreme'][year]:
                if word in fin_dict['supreme'][year][case]['words'].keys():
                    doc_freq[word] += 1

        for court in fin_dict['circuit']:
            for year in fin_dict['circuit'][court]:
                for case in fin_dict['circuit'][court][year]:
                    if word in fin_dict['circuit'][court][year][case]['words'].keys():
                        doc_freq[word] += 1

    f = open('/nlp/data/romap/law/task_1/data/files/testing/doc_freq.json', 'w+')
    f.write(json.dumps(doc_freq))'''


    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court_num + '/doc_freq.json', 'w+')
    f.write(json.dumps(doc_freq))
    
    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court_num + '/doc_freq.json', 'r')
    for line in f: doc_freq = json.loads(line)

    tfidf = {'supreme': {}, 'circuit': {}}
    tfidf['circuit'][court] = {}
    for year in fin_dict['circuit'][court]:
        tfidf['circuit'][court][year] = {}
        for case in fin_dict['circuit'][court][year]:
            tfidf['circuit'][court][year][case] = {'words': {}}
            words = fin_dict['circuit'][court][year][case]['words']
            for word in words:
                tf = words[word]; df = doc_freq[word]
                
                idf = np.log((1.0*num_docs)/df)
                tfidf['circuit'][court][year][case]['words'][word] = tf*idf

    #supreme tf-idf
    '''tfidf = {'supreme': {}, 'circuit': {}}
    for year in fin_dict['supreme']:
        tfidf['supreme'][year] = {}
        for case in fin_dict['supreme'][year]:

            tfidf['supreme'][year][case] = {'words': {}}
            words = fin_dict['supreme'][year][case]['words']
            for word in words:
                tf = words[word]; df = doc_freq[word]
                
                idf = np.log((1.0*num_docs)/df)
                tfidf['supreme'][year][case]['words'][word] = tf*idf'''
            

    '''for court in fin_dict['circuit']:
        tfidf['circuit'][court] = {}
        for year in fin_dict['circuit'][court]:
            tfidf['circuit'][court][year] = {}
            for case in fin_dict['circuit'][court][year]:
                tfidf['circuit'][court][year][case] = {'words': {}}
                words = fin_dict['circuit'][court][year][case]['words']
                for word in words:
                    tf = words[word]; df = doc_freq[word]
                    idf = np.log((1.0*num_docs)/df)
                    tfidf['circuit'][court][year][case]['words'][word] = tf*idf'''

    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court_num + '/tfidf.json', 'w+')
    f.write(json.dumps(tfidf))

def remove_unicode(court_num):
    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court_num + '/tfidf.json', 'r')
    for line in f: dict = json.loads(line)

    #supreme
    '''fin_dict = {'supreme':{}}
    for year in dict['supreme']:
        flag = True
        if year == '2017': flag = False
        if flag is False: continue
        fin_dict['supreme'][year] = {}
        for case in dict['supreme'][year]:
            fin_dict['supreme'][year][case] = {'words': {}}
            words = dict['supreme'][year][case]['words']; temp = {}
            for word in words:
                flag = False
                for char in word:
                    if ord(char) > 128:
                        flag = True
                if flag is True: continue
                temp[word] = words[word]
            fin_dict['supreme'][year][case]['words'] = temp'''

    #circuit
    court = court_num + '_circuit'
    fin_dict = {'circuit': {court: {}}}
    for year in dict['circuit'][court]:
        flag = True
        if year == '2017': flag = False
        if flag is True: continue
        fin_dict['circuit'][court][year] = {}
        for case in dict['circuit'][court][year]:
            fin_dict['circuit'][court][year][case] = {'words': {}}
            words = dict['circuit'][court][year][case]['words']; temp = {}
            for word in words:
                flag = False
                for char in word:
                    if ord(char) > 128:
                        flag = True
                if flag is True: continue
                temp[word] = words[word]
            fin_dict['circuit'][court][year][case]['words'] = temp
    
    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court_num + '/test_tfidf_cleaned.json', 'w+')
    f.write(json.dumps(fin_dict))

    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court_num + '/test_cases.neg', 'w+')
    newf = open('/nlp/data/romap/law/task_1/data/files/testing/' + court_num + '/test_cases.txt', 'w+')
    #pos for spreme, neg for circuit
    '''f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court_num + '/train_tfidf_cleaned.json', 'w+')
    f.write(json.dumps(fin_dict))

    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court_num + '/train_cases.neg', 'w+')
    newf = open('/nlp/data/romap/law/task_1/data/files/testing/' + court_num + '/train_cases.txt', 'w+')'''

    for year in fin_dict['circuit'][court]:
        for case in fin_dict['circuit'][court][year]:
            words = fin_dict['circuit'][court][year][case]['words']
            sorted_x = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
            for item in sorted_x[:100]:
                f.write(item[0] + ' ')
                newf.write(item[0] + ' ')
            f.write('\n')
            newf.write('\n\n')

    '''for year in fin_dict['supreme']:
        for case in fin_dict['supreme'][year]:
            words = fin_dict['supreme'][year][case]['words']
            sorted_x = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
            for item in sorted_x[:100]:
                f.write(item[0] + ' ')
                newf.write(item[0] + ' ')
            f.write('\n')
            newf.write('\n\n')'''

def copy_words(index_dict, court):
    #supreme
    '''path = '/nlp/data/romap/law/task_1/data/train/words/positive/'
    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court + '/train_tfidf_cleaned.json', 'r')
    for line in f:
        dict = json.loads(line)

    for year in dict['supreme']:
        for case in dict['supreme'][year]:
            if case not in index_dict[year].keys(): continue
            index = index_dict[year][case]
            case_words = dict['supreme'][year][case]['words']
            top_words = sorted(case_words.items(), key=operator.itemgetter(1), reverse=True)
            f = open(path + index + '.txt', 'w+')
            for item in top_words:
                f.write(item[0] + '\n')

    path = '/nlp/data/romap/law/task_1/data/test/words/positive/'
    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court + '/test_tfidf_cleaned.json', 'r')
    for line in f:
        dict = json.loads(line)

    for year in dict['supreme']:
        for case in dict['supreme'][year]:
            if case not in index_dict[year].keys(): continue
            index = index_dict[year][case]
            case_words = dict['supreme'][year][case]['words']
            top_words = sorted(case_words.items(), key=operator.itemgetter(1), reverse=True)
            f = open(path + index + '.txt', 'w+')
            for item in top_words:
                f.write(item[0] + '\n')'''

    #circuit
    path = '/nlp/data/romap/law/task_1/data/train/words/negative/'
    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court + '/train_tfidf_cleaned.json', 'r')
    for line in f:
        dict = json.loads(line)

    court_name = court + '_circuit'
    for year in dict['circuit'][court_name]:
        for case in dict['circuit'][court_name][year]:
            if case not in index_dict[year].keys(): continue
            index = index_dict[year][case]
            case_words = dict['circuit'][court_name][year][case]['words']
            top_words = sorted(case_words.items(), key=operator.itemgetter(1), reverse=True)
            f = open(path + index + '.txt', 'w+')
            for item in top_words:
                f.write(item[0] + '\n')

    path = '/nlp/data/romap/law/task_1/data/test/words/negative/'
    f = open('/nlp/data/romap/law/task_1/data/files/testing/' + court + '/test_tfidf_cleaned.json', 'r')
    for line in f:
        dict = json.loads(line)

    for year in dict['circuit'][court_name]:
        for case in dict['circuit'][court_name][year]:
            if case not in index_dict[year].keys(): continue
            index = index_dict[year][case]
            case_words = dict['circuit'][court_name][year][case]['words']
            top_words = sorted(case_words.items(), key=operator.itemgetter(1), reverse=True)
            f = open(path + index + '.txt', 'w+')
            for item in top_words:
                f.write(item[0] + '\n')

    
        
    
if __name__ == '__main__':
    #get_metadata()
    #get doc_freq dict -- running!
    #get case metadata (titles) -- done!
    #ruun tfidf after doc_freq is created to get tfidf dict -- todo!
    
    #create .txt file for top 100 unigrams of each doc -- todo!


    
    #copy_data()
    #remove_unicode('supreme')
    num = sys.argv[1]

    f = open('/nlp/data/romap/law/task_1/data/index_dict.json', 'r')
    for line in f: index_dict = json.loads(line)
    courts = [str(i) for i in range(1, 12)]; courts.append('district')
    for court in courts:
        court_name = court + '_circuit'
        copy_words(index_dict['circuit'][court_name], court)
    

    
    #top_tfidf(100)
    #top_tfidf(50)





