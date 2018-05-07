import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import sys
import sklearn
from sklearn.metrics import confusion_matrix
import os

path = '/nlp/data/romap/law/task_5/models_2/'

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
    true_pos = 0.0
    for i in range(len(gold_mask)):
        if gold_mask[i] == pred_mask[i]: true_pos += 1.0
    return true_pos/len(gold_mask)

def get_fm(prec, recl):
    if prec <= 0 or recl <= 0: return 0.0
    return (2.0*prec*recl)/(prec+recl)

def evaluate(fold):
    dirpath = path + 'title_results/' + str(fold) + '/' + set_name + '/' + model_name + '/' 
    true, pred = [], []
    if model_name == 'lstm' or model_name == 'log':
        fname = dirpath + 'results-0.json'
        if os.path.isfile(fname) is False: return 0, 0, 0, 0, [[0, 0], [0, 0]]
        f = open(fname, 'r')
        for line in f:
            temp = json.loads(line)

        for sent_id in temp:
            true.extend(temp[sent_id]['true'])
            pred.extend(temp[sent_id]['pred'])
            
    if model_name == 'svm':
        fname = dirpath + 'results-0.txt'
        if os.path.isfile(fname) is False: return 0, 0, 0, 0, [[0, 0], [0, 0]]
        f = open(fname, 'r')
        true, pred = [], []
        for line in f:
            items = line.strip().split(' ')
            pred.append(int(items[1])); true.append(int(items[0]))

    matrix = confusion_matrix(true, pred)
    tp, tn, fp, fn = matrix[0][0], matrix[1][1], matrix[0][1], matrix[1][0]
    total, pred_neg, pred_pos = tp+tn+fp+fn, tp+fn, tp+fp

    if tp == 0:
        prec = 0; recl = 0
    else:
        prec = float(tp)/pred_pos; recl = float(tp)/pred_neg
    acc = float(tp+tn)/total
    fm = get_fm(prec, recl)
    
    return round(prec, 2), round(recl, 2), round(acc, 2), round(fm, 2), matrix

def compile_results():
    models = ['models_1', 'models_2', 'models_spec_1', 'models_spec_2']
    f1 = open('/nlp/data/romap/law/results.txt', 'w+')
    wvecs = ['concept', 'google', 'legal']; fnames = ['results-log.txt', 'results-svm.txt', 'results-lstm.txt']
    vals = []
    for model in models:
        print model
        for fname in fnames:
            vals = [fname.split('.')[0].split('-')[-1]]
            for wvec in wvecs:
                print wvec
                path = '/nlp/data/romap/law/task_4/' + model + '/shell/' + wvec + '/results/';
                f = open(path + fname, 'r'); lines = f.readlines();
                if len(lines) > 2:
                    items = lines[1].strip().split('\t'); items = [round(float(item), 2) for item in items]
                else: items = [0, 0, 0, 0, 0]
                vals.extend(items[1:])
            for val in vals:
                f1.write(str(val) + '\t')
            f1.write('\n'); vals = []
        f1.write('\n\n')
    
if __name__ == '__main__':
    
    global set_name, model_name
    set_name = sys.argv[1]; model_name = sys.argv[2]

    #f = open(path + set_name + '/' + 'results/results-' + model_name + '.txt', 'w+')
    f = open('/nlp/data/romap/law/task_5/models_2/results_' + model_name + '.txt', 'w+')
    f.write('fold\tprecision\trecall\taccuracy\tfmeasure\n')
    prec, recl, acc, fm = [], [], [], []
    matrix = np.zeros((2, 2))
    #titles = 1 to 55
    for fold in range(1, 55):
        v1, v2, v3, v4, m = evaluate(fold); items = [v1, v2, v3, v4]
        f.write(str(fold) + '\t' + str(items[0]) + '\t' + str(items[1]) + '\t' + str(items[2]) + '\t' + str(items[3]) + '\n')
        prec.append(items[0]); recl.append(items[1]); acc.append(items[2]); fm.append(items[3])
        matrix += m
    f.write('\n\nmean\t' + str(round(np.mean(prec), 2)) + '\t' + str(round(np.mean(recl), 2)) + '\t' + str(round(np.mean(acc), 2)) )
    f.write('\noverall confusion:\n')
    f.write(str(matrix))
    
    #compile_results()

    
