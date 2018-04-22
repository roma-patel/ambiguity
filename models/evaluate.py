import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import sys

path = '/nlp/data/romap/law/task_3/models/shell/'
tags = {'D': 0, 'G': 1, 'O': 2}

def mask2spans(mask):
    spans = []; 
    if mask[0] == 1:
        sidx = 0
    for idx, v in enumerate(mask[1:], 1):
        # start of span
        if v==1 and mask[idx-1] == 0: 
            sidx = idx
        # end of span
        elif v==0 and mask[idx-1] == 1: 
            eidx = idx
            spans.append( (sidx, eidx) )
    return spans

def get_precision(true, pred):
    spans = mask2spans(true); mask = pred

    precision_arr = []
    for span in spans:
        length = span[1]-span[0]
        poss = sum(true[span[0]:span[1]])
        precision_arr.append(1.0*poss / length)
    precision = np.mean(precision_arr)

    return precision

def get_recall(true, pred):
    gold_spans = mask2spans(true); mask = pred
    recall_arr = []
    for span in gold_spans:
        length = span[1]-span[0]
        poss = sum(mask[span[0]:span[1]])
        recall_arr.append(1.0*poss / length)
    recall = np.mean(recall_arr)

    return recall

def get_accuracy(gold_mask, pred_mask):
    true_pos = 0
    for i in range(len(gold_mask)):
        if gold_mask[i] == pred_mask[i]: true_pos += 1
    return 1.0*true_pos/len(gold_mask)

def evaluate(fold):
    dirpath = path + set_name + '/' + 'results/' + model_name + '/'
    f = open(dirpath + 'results-' + str(fold) + '.json', 'r')
    for line in f: temp = json.loads(line)

    '''
    temp = {'1': {'tokens': [], 'true': [1, 1, 1], 'pred': [0, 0, 1]},
            '2': {'tokens': [], 'true': [1, 0, 1], 'pred': [0, 0, 1]},
            '3': {'tokens': [], 'true': [0, 1, 1], 'pred': [0, 0, 1]}
        }
    '''
    items = [[temp[key]['tokens'], temp[key]['true'], temp[key]['pred']] for key in temp.keys()]
    prec, recall, acc = [], [], []
    for item in items:
        true, pred = item[1], item[2]
        print len(true); print len(pred)
        true.append(0); pred.append(0)

        print true; print pred; print '\n'
        # three-class accuracy
        acc.append(get_accuracy(true, pred))
        
        # precision and recall of definition class
        true = [1 if item > 0 else 0 for item in true]
        pred = [1 if item > 0 else 0 for item in pred]
        prec.append(get_precision(true, pred))
        recall.append(get_recall(true, pred))

    return [round(np.mean(prec), 2), round(np.mean(recl), 2), round(np.mean(acc), 2)]


             
if __name__ == '__main__':
    global set_name, model_name
    set_name = sys.argv[1]; model_name = sys.argv[2]


    f = open(path + set_name + '/' + 'results/results-' + model_name + '.txt', 'w+')
    f.write('fold\tprecision\recall\accuracy\n')
    prec, recl, acc = [], [], []
    for fold in range(0, 10):
        items = evaluate(fold)
        f.write(str(fold) + '\t' + str(items[0]) + '\t' + str(items[1]) + '\t' + str(items[2]) + '\n')
        prec.append(items[0]); recl.append(items[1]); acc.append(items[2])

    f.write('\n\nmean\t' + str(round(np.mean(prec), 2)) + '\t' + str(round(np.mean(recl), 2)) + '\t' + str(round(np.mean(acc), 2)) )





    
