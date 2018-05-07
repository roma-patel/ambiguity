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
    true_pos = 0.0
    for i in range(len(gold_mask)):
        if gold_mask[i] == pred_mask[i]: true_pos += 1.0
    return true_pos/len(gold_mask)

def get_fm(prec, recl):
    return (2.0*prec*recl)/(prec+recl)

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

        '''
        print len(true); print len(pred)
        true.append(0); pred.append(0)
        # three-class accuracy
        acc.append(get_accuracy(true, pred))

        # precision and recall of definition class
        true = [1 if item > 0 else 0 for item in true]
        pred = [1 if item > 0 else 0 for item in pred]

        # check
        prec.append(get_precision(true, pred))
        recall.append(get_recall(true, pred))
        '''
        matrix = confusion_matrix(true, pred)
        row_sum = [np.sum(row) for row in matrix]
        col_sum = np.zeros(len(matrix))
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                col_sum[j] += matrix[i][j]

        acc.append(get_accuracy(true, pred))
        # definitions
        true_pos = 1.0*matrix[0][0];
        pred_pos = np.sum(row_sum[0]); pred_neg = np.sum(col_sum[0])

        # gloss
        '''
        true_pos = 1.0*matrix[1][1];
        pred_pos = np.sum(row_sum[1]); pred_neg = np.sum(col_sum[1])
        '''
        prec.append(true_pos/pred_pos)
        recall.append(true_pos/pred_neg)


    return [round(np.mean(prec), 2), round(np.mean(recall), 2), round(np.mean(acc), 2)]

def evaluate_spec(fold, collapse, invert):
    path = '/nlp/data/romap/law/task_3/models_spec/shell/'

    dirpath = path + set_name + '/' + 'results/' + model_name + '/'

    if model_name == 'lstm' or model_name == 'log':
        f = open(dirpath + 'results-' + str(fold) + '.json', 'r')
        for line in f: temp = json.loads(line)
        true, pred, tokens = temp['0']['true'], temp['0']['pred'], []

    if model_name == 'svm':
        f = open(dirpath + 'results-' + str(fold) + '.txt', 'r')
        true, pred = [], []
        for line in f:
            items = line.strip().split(' ')
            pred.append(int(items[0])); true.append(int(items[1]))
                    
    f = open('/nlp/data/romap/law/task_3/data/spec.json', 'r')
    for line in f: temp = json.loads(line)
    tokens = []
    sents = [temp[key]['tokens'] for key in temp]
    for sent in sents: tokens.extend([word.lower() for word in sent[0]])
    
    prec, recall, acc = [], [], []

    #collapse {0, 1} and 2
    if collapse == 1:
        pred = [0 if item < 2 else 1 for item in pred]

    #collapse {0, 2} and 1
    if collapse == 2:
        pred = [0 if item == 2 else item for item in pred]

    #collapse {1, 2} and 0
    if collapse == 3:
        pred = [0 if item == 0 else 1 for item in pred]

    # majority class baseline
    if collapse == 4:
        pred = [0 for item in pred]

        

    matrix = confusion_matrix(true, pred)
    tp, tn, fp, fn = matrix[0][0], matrix[1][1], matrix[0][1], matrix[1][0]

    if invert is True:
        tp, tn, fp, fn = matrix[1][1], matrix[0][0], matrix[1][0], matrix[0][1]

    total = tp+tn+fp+fn
    acc = 1.0*(tp+tn)/total
    if tp+fp == 0: prec = 1.0
    else:
        prec = 1.0*(tp)/(tp+fp)
    if tp+fn == 0: recl = 1.0
    else:
        recl = 1.0*(tp)/(tp+fn)


    acc = 1.0*(tp+tn)/total
    if tp+fp == 0: prec = 1.0
    else:
        prec = 1.0*(tp)/(tp+fp)
    if tp+fn == 0: recl = 1.0
    else:
        recl = 1.0*(tp)/(tp+fn)

    fm = get_fm(prec, recl)
    return [prec, recl, acc, fm]

   
    
             
if __name__ == '__main__':
    global set_name, model_name
    set_name = sys.argv[1]; model_name = sys.argv[2]
    #f = open(path + set_name + '/' + 'results/results-' + model_name + '.txt', 'w+')


    collapse = 1
    # change for spec
    path = '/nlp/data/romap/law/task_3/models_spec/shell/'

    f = open(path + set_name + '/' + 'results/results-' + model_name + '-' + str(collapse) + '.txt', 'w+')
    f.write('fold\tprecision\recall\accuracy\n')
    prec, recl, acc, fm = [], [], [], []
    prec_rand, recl_rand, acc_rand, fm_rand = [], [], [], []
    prec_inv, recl_inv, acc_inv, fm_inv = [], [], [], []
    
    for fold in range(0, 10):
        #items = evaluate(fold)
        items = evaluate_spec(fold, 1, False)
        f.write(str(fold) + '\t' + str(items[0]) + '\t' + str(items[1]) + '\t' + str(items[2]) + '\n')
        prec.append(items[0]); recl.append(items[1]); acc.append(items[2]); fm.append(items[3])

        items = evaluate_spec(fold, 4, False)
        prec_rand.append(items[0]); recl_rand.append(items[1]); acc_rand.append(items[2]); fm_rand.append(items[3])

        items = evaluate_spec(fold, collapse, True)
        prec_inv.append(items[0]); recl_inv.append(items[1]); acc_inv.append(items[2]); fm_inv.append(items[3])
        f.write(str(fold) + '-\t' + str(items[0]) + '\t' + str(items[1]) + '\t' + str(items[2]) + '\n')

    f.write('\n\nmean\t' + str(round(np.mean(prec), 2)) + '\t' + str(round(np.mean(recl), 2)) + '\t' + str(round(np.mean(acc), 2)) + '\t' + str(round(np.mean(fm), 2)) + '\n' )
    f.write('\n\nrand_mean\t' + str(round(np.mean(prec_rand), 2)) + '\t' + str(round(np.mean(recl_rand), 2)) + '\t' + str(round(np.mean(acc_rand), 2)) + '\t' + str(round(np.mean(fm_rand), 2)) + '\n' )
    f.write('\n\ninv_mean\t' + str(round(np.mean(prec_inv), 2)) + '\t' + str(round(np.mean(recl_inv), 2)) + '\t' + str(round(np.mean(acc_inv), 2)) + '\t' + str(round(np.mean(fm_inv), 2)) +'\n' )

    f.write('collapse {0, 1} and 2\n')

    collapse = 2
    # change for spec
    f = open(path + set_name + '/' + 'results/results-' + model_name + '-' + str(collapse) + '.txt', 'w+')
    f.write('fold\tprecision\recall\accuracy\n')
    prec, recl, acc, fm = [], [], [], []
    prec_rand, recl_rand, acc_rand, fm_rand = [], [], [], []
    prec_inv, recl_inv, acc_inv, fm_inv = [], [], [], []

    for fold in range(0, 10):
        #items = evaluate(fold)
        items = evaluate_spec(fold, collapse, False)
        f.write(str(fold) + '\t' + str(items[0]) + '\t' + str(items[1]) + '\t' + str(items[2]) + '\n')
        prec.append(items[0]); recl.append(items[1]); acc.append(items[2]); fm.append(items[3])

        items = evaluate_spec(fold, 4, False)
        prec_rand.append(items[0]); recl_rand.append(items[1]); acc_rand.append(items[2]); fm_rand.append(items[3])

        items = evaluate_spec(fold, collapse, True)
        prec_inv.append(items[0]); recl_inv.append(items[1]); acc_inv.append(items[2]); fm_inv.append(items[3])
        f.write(str(fold) + '-\t' + str(items[0]) + '\t' + str(items[1]) + '\t' + str(items[2]) + '\n')

    f.write('\n\nmean\t' + str(round(np.mean(prec), 2)) + '\t' + str(round(np.mean(recl), 2)) + '\t' + str(round(np.mean(acc), 2)) + '\t' + str(round(np.mean(fm), 2)) + '\n' )
    f.write('\n\nrand_mean\t' + str(round(np.mean(prec_rand), 2)) + '\t' + str(round(np.mean(recl_rand), 2)) + '\t' + str(round(np.mean(acc_rand), 2)) + '\t' + str(round(np.mean(fm_rand), 2)) + '\n' )
    f.write('\n\ninv_mean\t' + str(round(np.mean(prec_inv), 2)) + '\t' + str(round(np.mean(recl_inv), 2)) + '\t' + str(round(np.mean(acc_inv), 2)) + '\t' + str(round(np.mean(fm_inv), 2)) +'\n' )

    f.write('collapse {0, 2} and 1\n')




    collapse = 3
    # change for spec
    f = open(path + set_name + '/' + 'results/results-' + model_name + '-' + str(collapse) + '.txt', 'w+')
    f.write('fold\tprecision\trecall\taccuracy\n')
    prec, recl, acc, fm = [], [], [], []
    prec_rand, recl_rand, acc_rand, fm_rand = [], [], [], []
    prec_inv, recl_inv, acc_inv, fm_inv = [], [], [], []

    for fold in range(0, 10):
        #items = evaluate(fold)
        items = evaluate_spec(fold, collapse, False)
        f.write(str(fold) + '\t' + str(items[0]) + '\t' + str(items[1]) + '\t' + str(items[2]) + '\n')
        prec.append(items[0]); recl.append(items[1]); acc.append(items[2]); fm.append(items[3])

        items = evaluate_spec(fold, 4, False)
        prec_rand.append(items[0]); recl_rand.append(items[1]); acc_rand.append(items[2]); fm_rand.append(items[3])

        items = evaluate_spec(fold, collapse, True)
        prec_inv.append(items[0]); recl_inv.append(items[1]); acc_inv.append(items[2]); fm_inv.append(items[3])
        f.write(str(fold) + '-\t' + str(items[0]) + '\t' + str(items[1]) + '\t' + str(items[2]) + '\n')

    f.write('\n\nmean\t' + str(round(np.mean(prec), 2)) + '\t' + str(round(np.mean(recl), 2)) + '\t' + str(round(np.mean(acc), 2)) + '\t' + str(round(np.mean(fm), 2)) + '\n' )
    f.write('\n\nrand_mean\t' + str(round(np.mean(prec_rand), 2)) + '\t' + str(round(np.mean(recl_rand), 2)) + '\t' + str(round(np.mean(acc_rand), 2)) + '\t' + str(round(np.mean(fm_rand), 2)) + '\n' )
    f.write('\n\ninv_mean\t' + str(round(np.mean(prec_inv), 2)) + '\t' + str(round(np.mean(recl_inv), 2)) + '\t' + str(round(np.mean(acc_inv), 2)) + '\t' + str(round(np.mean(fm_inv), 2)) +'\n' )

    f.write('collapse {1, 2} and 0\n')





    
