import json
import numpy as np
import re
from collections import Counter
       
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


def mrc():
    stopwords = []
    f = open('/nlp/data/romap/law/data/stopwords.txt', 'r')
    for line in f: stopwords.append(line.strip())
    
    data, ratings = {}, {}
    '''
    f = open('/nlp/data/romap/law/concreteness.txt', 'r')
    lines = f.readlines()
    for line in lines[1:]:
        items = line.strip().split('\t')
        ratings[items[0]] = float(items[2])
    
    f = open('/nlp/data/romap/law/mrc.tsv', 'r')
    lines = f.readlines()
    for line in lines:
        items = line.strip().split('\t')
        ratings[items[0]] = float(items[1])
    '''
    f = open('/nlp/data/romap/law/ratings.json', 'r')
    for line in f: ratings = json.loads(line)


    
    f = open('/nlp/data/romap/law/dgo_titles_cleaned.json', 'r')
    for line in f:
        temp = json.loads(line)
        data[temp['title']] = temp

    f = open('/nlp/data/romap/law/std-distribution.csv', 'w+')
    #f = open('/nlp/data/romap/law/mrc-distribution.csv', 'w+')

    f.write('title\tdef\tgloss\tother\tset(def)\tset(gloss)\tset(other)\td_mrc\tg_mrc\to_mrc\tset(d_mrc)\tset(g_mrc)\tset(o_mrc)\td_wavg\tg_wavg\to_wavg\td_avg\tg_avg\to_avg\td_wavg\tg_wavg\to_wavg\td_avg\tg_avg\to_avg\n')

    fin = {}
    words = [word for word in ratings.keys() if len(ratings[word]['conc']) > 0 and len(ratings[word]['mrc']) > 0]
    for title in data:
        items = []
        if 'definition' not in data[title].keys(): continue

        definitions, gloss, other = [word for word in data[title]['definition'] if word not in stopwords], [word for word in data[title]['gloss'] if word not in stopwords], [word for word in data[title]['other'] if word not in stopwords]
        dc, gc, oc = Counter(definitions), Counter(gloss), Counter(other)
        items = [title]
        items.extend([len(definitions), len(gloss), len(other), len(dc.keys()), len(gc.keys()), len(oc.keys())])

        # number of unique words
        dconc = list(set(words) & set(definitions))
        gconc = list(set(words) & set(gloss))
        oconc = list(set(words) & set(other))

        # number of words with weights
        dcount = np.sum([dc[word] for word in dconc])
        gcount = np.sum([gc[word] for word in gconc])
        ocount = np.sum([oc[word] for word in oconc])

        #items.extend([dcount, gcount, ocount, len(dconc), len(gconc), len(oconc)])

        #conc
        # average score
        dscore = (1.0*np.sum([ratings[word]['conc'][0] for word in dconc]))/len(dconc)
        gscore = (1.0*np.sum([ratings[word]['conc'][0] for word in gconc]))/len(gconc)
        oscore = (1.0*np.sum([ratings[word]['conc'][0] for word in oconc]))/len(oconc)

        # weighted average score
        dwscore = (1.0*np.sum([dc[word]*ratings[word]['conc'][0] for word in dconc]))/dcount
        gwscore = (1.0*np.sum([gc[word]*ratings[word]['conc'][0] for word in gconc]))/gcount
        owscore = (1.0*np.sum([oc[word]*ratings[word]['conc'][0] for word in oconc]))/ocount

        #items.extend([round(dscore, 2), round(gscore, 2), round(oscore, 2), round(dwscore, 2), round(gwscore, 2), round(owscore, 2)])
        items.extend([round(dwscore, 2), round(gwscore, 2), round(owscore, 2)])

        d = [[ratings[word]['conc'][0] for i in range(0, dc[word])] for word in dconc]
        g = [[ratings[word]['conc'][0] for i in range(0, gc[word])] for word in gconc]
        o = [[ratings[word]['conc'][0] for i in range(0, oc[word])] for word in oconc]

        d = [item for sublist in d for item in sublist]
        g = [item for sublist in g for item in sublist]
        o = [item for sublist in o for item in sublist]

        items.extend([np.std(d), np.std(g), np.std(o)])
        fin[title] = {'definition': d, 'gloss': g, 'other': o}

        dscore = (1.0*np.sum([ratings[word]['mrc'][0] for word in dconc]))/len(dconc)
        gscore = (1.0*np.sum([ratings[word]['mrc'][0] for word in gconc]))/len(gconc)
        oscore = (1.0*np.sum([ratings[word]['mrc'][0] for word in oconc]))/len(oconc)

        # weighted average score
        dwscore = (1.0*np.sum([dc[word]*ratings[word]['mrc'][0] for word in dconc]))/dcount
        gwscore = (1.0*np.sum([gc[word]*ratings[word]['mrc'][0] for word in gconc]))/gcount
        owscore = (1.0*np.sum([oc[word]*ratings[word]['mrc'][0] for word in oconc]))/ocount

        #items.extend([round(dscore, 2), round(gscore, 2), round(oscore, 2), round(dwscore, 2), round(gwscore, 2), round(owscore, 2)])
        items.extend([round(dwscore, 2), round(gwscore, 2), round(owscore, 2)])

        d = [[ratings[word]['mrc'][0] for i in range(0, dc[word])] for word in dconc]
        g = [[ratings[word]['mrc'][0] for i in range(0, gc[word])] for word in gconc]
        o = [[ratings[word]['mrc'][0] for i in range(0, oc[word])] for word in oconc]

        d = [item for sublist in d for item in sublist]
        g = [item for sublist in g for item in sublist]
        o = [item for sublist in o for item in sublist]

        items.extend([np.std(d), np.std(g), np.std(o)])
        
        for item in items:
            f.write(str(item) + '\t')
        f.write('\n')

    f = open('/nlp/data/romap/law/dist.json', 'w+')
    f.write(json.dumps(fin))
    
if __name__ == '__main__':
    mrc()
    #get_distribution()
    #get_sentences()
   # break_unigrams()
    #run()
