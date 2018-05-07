import json
import numpy as np
import os, sys
import re

path = '/nlp/data/romap/law/task_5/'

def get_speeches():
    data_path = '/nlp/data/romap/random_datasets/president-speeches/'
    filenames = os.listdir(data_path)
    names = ['new-trump', 'new-clinton', 'obama', 'bush', 'clinton', 'reagan', 'carter', 'ford', 'nixon']
    speeches = {}
    for name in names:
        speeches[name] = {}
        for filename in os.listdir(data_path + name + '/'):
            if filename.endswith('txt') is False: continue
            f = open(data_path + name + '/' + filename, 'r')
            lines = f.readlines()
            speeches[name][str(len(speeches[name]))] = [line.strip() for line in lines]
    f = open('/nlp/data/romap/random_datasets/president-speeches/speeches.json', 'w+')
    f.write(json.dumps(speeches))


    
def test():
    speeches = get_speeches()
if __name__ == '__main__':
    test()




    
