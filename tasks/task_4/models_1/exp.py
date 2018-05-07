import json
import numpy as np
import re
from collections import Counter
import os

def get_distribution():
    
    data, concept, google = {}, [], []
    f = open('/nlp/data/romap/law/dgo_titles_cleaned.json', 'r')
    for line in f:
        temp = json.loads(line)
        data[temp['title']] = temp

    f = open('/nlp/data/romap/law/task_2/files/conceptnet.txt', 'r')
    for line in f: concept.append(line.strip())

    f = open('/nlp/data/romap/law/task_2/files/google_news.txt', 'r')
    for line in f: google.append(line.strip())

    temp = {}
    for title in data:
        definitions, gloss, other = Counter(data[title]['definition']), Counter(data[title]['gloss']), Counter(data[title]['other'])
        set_dc = list(set(definitions.keys()) & set(concept))
        set_gc = list(set(gloss.keys()) & set(concept))
        set_oc = list(set(other.keys()) & set(concept))
        set_dg = list(set(definitions.keys()) & set(google))
        set_gg = list(set(gloss.keys()) & set(google))
        set_og = list(set(other.keys()) & set(google))

        temp[title] = {'definition': definitions, 'gloss': gloss, 'other': other, 'set_dc': set_dc, 'set_gc': set_gc, 'set_oc': set_oc, 'set_dg': set_dg, 'set_gg': set_gg, 'set_og': set_og} 

        '''
        set_dc = Counter([word for word in definitions.keys() if word in concept])
        set_dg = Counter([word for word in definitions.keys() if word in google])
        set_gc = Counter([word for word in gloss.keys() if word in concept])
        set_gg = Counter([word for word in gloss.keys() if word in google])
        set_oc = Counter([word for word in other.keys() if word in concept])
        set_og = Counter([word for word in other.keys() if word in google])

        s = title + ',' + str(sum(definitions.values())) + ',' + str(sum(gloss.values())) + ',' + str(sum(other.values())) + ',' + str(len(definitions.keys())) + ',' + str(len(gloss.keys())) + ',' + str(len(other.keys())) + ','
        s += str(sum(set_dc.values())) + ',' + str(sum(set_gc.values())) + ',' + str(sum(set_oc.values())) + ',' + str(len(set_dc.keys())) + ',' + str(len(set_gc.keys())) + ',' + str(len(set_oc.keys())) + ','
        s += str(sum(set_dg.values())) + ',' + str(sum(set_gg.values())) + ',' + str(sum(set_og.values())) + ',' + str(len(set_dg.keys())) + ',' + str(len(set_gg.keys())) + ',' + str(len(set_og.keys()))

        f.write(s + '\n')
        '''
    f = open('/nlp/data/romap/law/distribution.json', 'w+')
    for title in temp:
        t = temp[title]; t['title'] = title
        f.write(json.dumps(t)); f.write('\n')

    

    data = {}
    f = open('/nlp/data/romap/law/distribution.json', 'r')
    for line in f:
        temp = json.loads(line)
        data[temp['title']] = temp

    f = open('/nlp/data/romap/law/distribution.csv', 'w+')

    for title in data:
        if 'definition' not in data[title].keys(): continue
        definitions, gloss, other = data[title]['definition'], data[title]['gloss'], data[title]['other']
        s = title + ',' + str(sum(definitions.values())) + ',' + str(sum(gloss.values())) + ',' + str(sum(other.values())) + ',' + str(len(definitions.keys())) + ',' + str(len(gloss.keys())) + ',' + str(len(other.keys())) + ','

        def_concept = sum([definitions[word] for word in data[title]['set_dc']])
        gloss_concept = sum([gloss[word] for word in data[title]['set_gc']])
        other_concept = sum([other[word] for word in data[title]['set_oc']])

        s += str(def_concept) + ',' + str(gloss_concept) + ',' + str(other_concept) + ',' + str(len(data[title]['set_dc'])) + ',' + str(len(data[title]['set_gc'])) + ',' + str(len(data[title]['set_oc'])) + ','

        def_google = sum([definitions[word] for word in data[title]['set_dg']])
        gloss_google = sum([gloss[word] for word in data[title]['set_gg']])
        other_google = sum([other[word] for word in data[title]['set_og']])

        s += str(def_google) + ',' + str(gloss_google) + ',' + str(other_google) + ',' + str(len(data[title]['set_dg'])) + ',' + str(len(data[title]['set_gg'])) + ',' + str(len(data[title]['set_og'])) 
        f.write(s + '\n')

    #for each def [#sections_def, #sections_gloss, #sections_other]

def break_unigrams():

    '''
    words, sentences = {}, {}
    f = open('/nlp/data/romap/law/dgo_titles_cleaned.json', 'r')
    for line in f:
        temp = json.loads(line)
        words[temp['title']] = temp

    f = open('/nlp/data/romap/law/title_sentences.json', 'r')
    for line in f:
        temp = json.loads(line)
        sentences[temp['title']] = temp


    f = open('/nlp/data/romap/law/word_labels.tsv', 'w+')
    f.write('word\tlabel\ttitle_no\tsection_no\tsentence_id\n')
    for title in sentences:
        if 'definition' not in words[title].keys():
            definitions, gloss, other = [], [], []
        else:
            definitions, gloss, other = words[title]['definition'], words[title]['gloss'], words[title]['other']

        for section in sentences[title]:
            for sent_id in sentences[title][section]:
                for word in sentences[title][section][sent_id]:
                    if word in definitions: label = 'D'
                    elif word in gloss: label = 'G'
                    else: label = 'O'
                    f.write(word + '\t' + label + '\t' + title + '\t' + section + '\t' + str(sent_id) + '\n')
                f.write('\n')
    '''
    concept, google, all_words = [], [], []
    f = open('/nlp/data/romap/law/task_2/files/conceptnet.txt', 'r')
    for line in f: concept.append(line.strip())

    f = open('/nlp/data/romap/law/task_2/files/google_news.txt', 'r')
    for line in f: google.append(line.strip())

    words, sentences = {}, {}
    f = open('/nlp/data/romap/law/dgo_titles_cleaned.json', 'r')
    for line in f:
        temp = json.loads(line)
        words[temp['title']] = temp
        if 'definition' in temp.keys():
            all_words.extend(temp['definition'])
            all_words.extend(temp['gloss'])
            all_words.extend(temp['other'])

    all_words = list(set(all_words)); cset = list(set(all_words) & set(concept))
    gset = list(set(all_words) & set(google))

    f = open('/nlp/data/romap/law/title_sentences.json', 'r')
    for line in f:
        temp = json.loads(line)
        sentences[temp['title']] = temp


    f1 = open('/nlp/data/romap/law/word_labels_concept.tsv', 'w+')
    f2 = open('/nlp/data/romap/law/word_labels_google.tsv', 'w+')

    f1.write('word\tlabel\ttitle_no\tsection_no\tsentence_id\n')
    f1.write('word\tlabel\ttitle_no\tsection_no\tsentence_id\n')

    for title in sentences:
        if 'definition' not in words[title].keys():
            definitions, gloss, other = [], [], []
        else:
            definitions, gloss, other = words[title]['definition'], words[title]['gloss'], words[title]['other']

        for section in sentences[title]:
            for sent_id in sentences[title][section]:
                for word in sentences[title][section][sent_id]:
                    if word in definitions: label = 'D'
                    elif word in gloss: label = 'G'
                    else: label = 'O'
                    if word in cset:
                        f1.write(word + '\t' + label + '\t' + title + '\t' + section + '\t' + str(sent_id) + '\n')
                    if word in gset:
                        f2.write(word + '\t' + label + '\t' + title + '\t' + section + '\t' + str(sent_id) + '\n')

                f1.write('\n'); f2.write('\n')


# takes in a list of strings, returns list of tokenised strings       
def clean_sentences(sentences):
    for i in range(len(sentences)):
        sentence = re.sub('[,:;\(\)!@$%^&\[\]+=0-9]', ' ', sentences[i])
        words = [word.lower() for word in sentence.split(' ') if len(word) > 1 and '-' not in word]
        sentences[i] = words
        
    return sentences


def get_sentences():
    # dictionaries = tlsd.json and dgo.json
    # iterate through sections in a title in tlsd, clean, sentence split
    # and tokenise
    # preserve sentence order and take only words in dgo[title]
    # store [word, label, title-section, sent-id]

    '''
    f = open('/nlp/data/romap/law/titlesectionlinedict.json', 'r')
    for line in f: tlsd = json.loads(line)

    data = {}
    f = open('/nlp/data/romap/law/dgo_titles_cleaned.json', 'r')
    for line in f:
        temp = json.loads(line)
        data[temp['title']] = temp

    fin_dict = {}
    for title_name in tlsd:
        title = title_name.split(' - ')[0].split(' ')[-1]
        fin_dict[title] = {}
        for section in tlsd[title_name]:
            sdict = {}; text = tlsd[title_name][section]
            sentences = text.split('.')
            sentences = clean_sentences(sentences)
            for sentence in sentences:
                sdict[len(sdict.keys())] = sentence
                
            fin_dict[title][section] = sdict

    f = open('/nlp/data/romap/law/title_sentences.json', 'w+')
    for title in fin_dict:
        temp = fin_dict[title]; temp['title'] = title
        f.write(json.dumps(temp) + '\n')
    '''

    data = {}
    f = open('/nlp/data/romap/law/title_sentences.json', 'r')
    for line in f:
        temp = json.loads(line)
        data[temp['title']] = temp

    sentences = {}
    for title in data:
        sentences[title] = {}
        for section in data[title]:
            if section == 'title': continue
            temp = {}
            for sentid in data[title][section]:
                if len(data[title][section][sentid]) > 5:
                    temp[len(temp.keys())] = data[title][section][sentid]
            sentences[title][section] = temp

    f = open('/nlp/data/romap/law/title_sentences.json', 'w+')
    for title in sentences:
        temp = sentences[title]; temp['title'] = title
        f.write(json.dumps(temp) + '\n')

def test_folds():
    
    path = '/nlp/data/romap/law/task_4/data/'
    dist = {}
    for i in range(0, 10):
        train, test = path + str(i) + '/train.txt', path + str(i) + '/test.txt'
        f = open(train, 'r'); lines = f.readlines()
        temp = {}
        for line in lines:
            items = line.split('\t')
            word, tag, title = items[0], items[1], items[2]
            if title not in temp.keys():
                temp[title] = {'definition':[], 'gloss': [], 'other': []}
            if tag == 'D': temp[title]['definition'].append(word)
            if tag == 'G': temp[title]['gloss'].append(word)
            if tag == 'O': temp[title]['other'].append(word)
        for title in temp:
            temp[title]['definition'] = Counter(temp[title]['definition'])
            temp[title]['gloss'] = Counter(temp[title]['gloss'])
            temp[title]['other'] = Counter(temp[title]['other'])
        dist[str(i)] = temp

    f = open(path + 'words_train.json', 'w+')
    f.write(json.dumps(dist)); f.close()

    dist = {}
    for i in range(0, 10):
        train, test = path + str(i) + '/train.txt', path + str(i) + '/test.txt'
        f = open(test, 'r'); lines = f.readlines()
        temp = {}
        for line in lines:
            items = line.split('\t')
            word, tag, title = items[0], items[1], items[2]
            if title not in temp.keys():
                temp[title] = {'definition':[], 'gloss': [], 'other': []}
            if tag == 'D': temp[title]['definition'].append(word)
            if tag == 'G': temp[title]['gloss'].append(word)
            if tag == 'O': temp[title]['other'].append(word)
        for title in temp:
            temp[title]['definition'] = Counter(temp[title]['definition'])
            temp[title]['gloss'] = Counter(temp[title]['gloss'])
            temp[title]['other'] = Counter(temp[title]['other'])
        dist[str(i)] = temp

    f = open(path + 'words_test.json', 'w+')
    f.write(json.dumps(dist)); f.close()
    
    path = '/nlp/data/romap/law/task_4/data/'
    f = open(path + 'words_train.json', 'r')
    for line in f: dist = json.loads(line)

    for fold in dist:
        f = open(path + fold + '/distribution_train.txt', 'w+')
        for title in sorted(dist[fold].keys()):
            d, g, o = 0, 0, 0
            for word in dist[fold][title]['definition'].keys(): d += dist[fold][title]['definition'][word]
            for word in dist[fold][title]['gloss'].keys(): g += dist[fold][title]['gloss'][word]
            for word in dist[fold][title]['other'].keys(): o += dist[fold][title]['other'][word]

            f.write(str(title) + '\t' + str(d) + '\t' + str(g) + '\t' + str(o) + '\t' + str(len(dist[fold][title]['definition'].keys())) + '\t' + str(len(dist[fold][title]['gloss'].keys())) + '\t' + str(len(dist[fold][title]['other'].keys())) + '\n')
        f.close()

    f = open(path + 'words_test.json', 'r')
    for line in f: dist = json.loads(line)

    for fold in dist:
        f = open(path + fold + '/distribution_test.txt', 'w+')
        for title in sorted(dist[fold].keys()):
            d, g, o = 0, 0, 0
            for word in dist[fold][title]['definition'].keys(): d += dist[fold][title]['definition'][word]
            for word in dist[fold][title]['gloss'].keys(): g += dist[fold][title]['gloss'][word]
            for word in dist[fold][title]['other'].keys(): o += dist[fold][title]['other'][word]

            f.write(str(title) + '\t' + str(d) + '\t' + str(g) + '\t' + str(o) + '\t' + str(len(dist[fold][title]['definition'].keys())) + '\t' + str(len(dist[fold][title]['gloss'].keys())) + '\t' + str(len(dist[fold][title]['other'].keys())) + '\n')
        f.close()

def test():
    '''
    dist = {}
    f = open('/nlp/data/romap/law/word_labels_legal_3.tsv', 'r')
    lines = f.readlines()
    for line in lines[1:]:
        items = line.strip().split('\t')
        word, tag, title = items[0], items[1], items[2]
        if title not in dist.keys(): dist[title] = []
        dist[title].append(line)

    path = '/nlp/data/romap/law/task_4/data/titles/'
    for title in dist:
        if os.path.isdir(path + title) is False: os.mkdir(path+title)

        f = open(path + title + '/words.txt', 'w+')
        for line in dist[title]:
            f.write(line)
     '''

    #create test set -- sample from each title
    #create train set for each title, remove sampled words

    '''
    path = '/nlp/data/romap/law/task_4/data/titles/'
    for title in range(1, 54):
        if os.path.isdir(path + str(title)) is False: continue
        f = open(path + str(title) + '/words.txt', 'r')
        lines = f.readlines()
        test = lines[:100]; train = lines[100:]
        f = open(path + str(title) + '/test.txt', 'w+')
        for line in test:
            f.write(line)
        f.close()
        f = open(path + str(title) + '/train.txt', 'w+')
        for line in train:
            f.write(line)
        f.close()
    '''
    path = '/nlp/data/romap/law/task_4/data/titles/'
    all_train = []; all_test = []
    for title in range(1, 54):
        if os.path.isdir(path + str(title)) is False: continue
        f = open(path + str(title) + '/train.txt', 'r')
        lines = f.readlines()
        all_train.extend(lines)
        f = open(path + str(title) + '/test.txt', 'r')
        lines = f.readlines()
        all_test.extend(lines)

    f = open(path + 'all_test.txt', 'w+')
    for line in all_test: f.write(line)
    f.close()
    f = open(path + 'all_train.txt', 'w+')
    for line in all_train: f.write(line)
    f.close()
    


    
        
        
if __name__ == '__main__':
    test()
    #get_distribution()
    #get_sentences()
    #break_unigrams()
    #run()
