import json
import numpy as np
import re

embedding_dim, hidden_dim = 6, 6

def clean(text):
    text = re.sub('[.,:;\[\]\(\)?!/\"]', ' ', text)
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

    return [item for item in text.split(' ') if len(item) > 1]
       
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
        for section in data[title]:
            if section == 'title': continue
            if section not in text_dict[full_title].keys(): continue
            if len(data[title][section]['definition']) < 1: continue
            print section
            definitions, gloss, other = data[title][section]['definition'], data[title][section]['gloss'], data[title][section]['other']
            text = text_dict[full_title][section]
            sentences = sentence_split(text)
            sentences = [clean(item) for item in sentences]
            sentence_tokens = [tokenise(item) for item in sentences]
            temp[title][section] = sentence_tokens

    f = open('/nlp/data/romap/law/temp.json', 'w+')
    f.write(json.dumps(temp))
            

if __name__ == '__main__':
    run()
