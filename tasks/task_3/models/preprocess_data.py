import json
import numpy as np
import re

embedding_dim, hidden_dim = 6, 6

def clean(text):
    text = re.sub('[.,:;\[\]\(\)?!/\"0-9]', ' ', text)
    return text

def sentence_split(text):
    return text.split('.')

def lowercase(text):
    return text.lower()

def tokenise(text):
    
    stopwords = []
    f = open('/nlp/data/romap/law/data/stopwords.txt', 'r')
    for line in f: stopwords.append(line.strip())
    
    #remove stopwords
    #return [item for item in text.split(' ') if len(item) > 1 and item not in stopwords]

    return [item for item in text.split(' ') if len(item) > 1 and '-' not in item]
       
def run():
    data = {}
    f = open('/nlp/data/romap/law/dgo_sections.json', 'r')
    for line in f:
        temp = json.loads(line)
        data[temp['title']] = temp

    f = open('/nlp/data/romap/law/titlesectionlinedict.json', 'r')
    for line in f: text_dict = json.loads(line)

    temp = {}
    for full_title in text_dict:
        title = full_title.split(' - ')[0].split(' ')[-1]
        temp[title] = {}
        for section in text_dict[full_title]:
            #if len(data[title][section]['definition']) < 1: continue
            print section
            #definitions, gloss, other = data[title][section]['definition'], data[title][section]['gloss'], data[title][section]['other']
            text = text_dict[full_title][section]
            sentences = sentence_split(text)
            sentences = [clean(item) for item in sentences]
            sentence_tokens = [tokenise(item) for item in sentences]
            temp[title][section] = sentence_tokens

    f = open('/nlp/data/romap/law/temp_all.json', 'w+')
    f.write(json.dumps(temp))

def create_files():
    stopwords = []
    f = open('/nlp/data/romap/law/data/stopwords.txt', 'r')
    for line in f: stopwords.append(line.strip())
    
    data, concept, google = {}, [], []
    sections = {}

    f = open('/nlp/data/romap/law/dgo_sections.json', 'r')
    for line in f:
        temp = json.loads(line)
        sections[temp['title']] = temp
    
    f = open('/nlp/data/romap/law/dgo_titles_cleaned.json', 'r')
    for line in f:
        temp = json.loads(line)
        data[temp['title']] = temp

    f = open('/nlp/data/romap/law/task_2/files/conceptnet.txt', 'r')
    for line in f: concept.append(line.strip())

    f = open('/nlp/data/romap/law/task_2/files/google_news.txt', 'r')
    for line in f: google.append(line.strip())

    f = open('/nlp/data/romap/law/temp_all.json', 'r')
    for line in f:
        temp = json.loads(line)

    '''
    f1 = open('/nlp/data/romap/law/word_labels_concept_2.tsv', 'w+')
    f2 = open('/nlp/data/romap/law/word_labels_google_2.tsv', 'w+')
    f3 = open('/nlp/data/romap/law/word_labels_legal_2.tsv', 'w+')
    f1.write('word\tlabel\ttitle_no\tsection_no\tsentence_id\n')
    f2.write('word\tlabel\ttitle_no\tsection_no\tsentence_id\n')
    f3.write('word\tlabel\ttitle_no\tsection_no\tsentence_id\n')
    '''
    f1 = open('/nlp/data/romap/law/word_labels.tsv', 'w+')

    words = []; items = []
    for title in temp:
        print title
        definitions, gloss, other = list(set(data[title]['definition'])), list(set(data[title]['gloss'])), list(set(data[title]['other']))
        for section in temp[title]:
            if section not in sections[title].keys(): continue
            for sent_id in range(len(temp[title][section])):
                sentence = temp[title][section][sent_id]
                if len(sentence) < 5: continue
                for word in sentence:
                    word = word.lower()
                    if word in stopwords: continue
                    if word in definitions: label = 'D'
                    elif word in gloss: label = 'G'
                    else: label = 'O'
                    s = word + '\t' + label + '\t' + str(title) + '\t' + str(section) + '\t' + str(sent_id) + '\n'
                    f1.write(s)
                    words.append(word)
                    items.append(s)
                    '''
                    if word in concept: f1.write(s)
                    if word in google: f2.write(s)
                    f3.write(s)
                    '''

    c = list(set(concept)&set(words))
    g = list(set(google)&set(words))
    f1 = open('/nlp/data/romap/law/word_labels_concept_2.tsv', 'w+')
    f2 = open('/nlp/data/romap/law/word_labels_google_2.tsv', 'w+')
    f3 = open('/nlp/data/romap/law/word_labels_legal_2.tsv', 'w+')

    for i in range(len(items)):
        f3.write(items[i])
        if words[i] in c: f1.write(items[i])
        if words[i] in g: f2.write(items[i])



if __name__ == '__main__':
    create_files()
