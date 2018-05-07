import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as Functional
import torch.optim as optim
import json, sys, os, logging
import numpy as np
import argparse

import torchtext.datasets as datasets
import torchtext.data as data

embedding_dim = 300

class ConvNet(nn.Module):
    
    def __init__(self, args, embedding_dim, word_to_idx, tagset_size):
        super(ConvNet, self).__init__()

        self.args = args
        self.word_embeddings = nn.Embedding(len(word_to_idx), embedding_dim)
        #self.get_embeddings(word_to_idx, embedding_dim, wvec_paths[args.model_name])
        
        self.conv1 = nn.Conv2d(args.input_channel, args.output_channel, (1, 3))
        self.conv2 = nn.Conv2d(args.input_channel, args.output_channel, (1, 4))
        self.conv2 = nn.Conv2d(args.input_channel, args.output_channel, (1, 5))
        self.dropout = nn.Dropout(self.args.dropout)
        self.fc1 = nn.Linear(3*args.output_channel, tagset_size)

    def conv_and_pool(self, x, conv):
        x = Functional.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = Functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, x):
        x = self.word_embeddings(x)  # (N, W, D)
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [Functional.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [Functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.args.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

    def get_embeddings(self, word_to_idx, embedding_dim, wvec_path):
        word_vecs = {}; vocab = word_to_idx.keys()
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
                word, vec = items[0], [float(item) for item in items[1:]]
                if word in vocab: word_vecs[word] = vec
                
        
        weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
        mat = np.zeros((len(word_to_idx), embedding_dim))
        for word in word_to_idx:
            if word not in word_vecs: mat[word_to_idx[word]] = np.random.rand(1, 300)
            else: mat[word_to_idx[word]] = word_vecs[word]
        # add pretrained
        self.word_embeddings.weight.data.copy_(torch.from_numpy(mat))

        
        
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def to_scalar(var):
    return var.view(-1).data.tolist()[0]

def load_data():
    f = open('/nlp/data/romap/law/task_3/data/spec.json', 'r')
    for line in f: temp = json.loads(line)

    data = [(temp[key]['tokens'], temp[key]['tags']) for key in temp]
    cutoff = len(data) - int(0.1*len(data))
    train_data, test_data = data[:cutoff], data[cutoff:]

    return train_data, test_data

    
def prepare_data():
    training_data = [
    ("It just depends on how we deploy the child".split(), ['I', 'O', 'O', 'I', 'O', 'O', 'I', 'O', 'I']),
    ("The term child means young adult .".split(), ['O', 'O', 'I', 'O', 'O', 'O', 'O']),
     ("The child was injured .".split(), ['O', 'I', 'O', 'O', 'O'])
    ]

    testing_data = [
    ("The term child means person above 18 years .".split(),
             ["O", "O", "I", "O", "O", "O", "O", "O", "O"])
    ]
    word_to_idx = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    for sent, tags in testing_data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
                
    print 'word_to_idx'
    print(word_to_idx)
    tag_to_idx = {"I": 0, "O": 1, "D": 2}
    return training_data, testing_data, word_to_idx, tag_to_idx



    
    '''
    training_data, testing_data = load_data()
    word_to_idx = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    for sent, tags in testing_data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
                
    tag_to_idx = {"I": 0, "O": 1}
    
    return training_data, testing_data, word_to_idx, tag_to_idx
    '''
    
def train():
    args = get_args()
    training_data, testing_data, word_to_idx, tag_to_idx = prepare_data()
    model = ConvNet(args, embedding_dim, word_to_idx, len(tag_to_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    text_field = data.Field(lower=True); label_field = data.Field(sequential=False)
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args.batch_size, 
                                                     len(dev_data), 
                                                     len(test_data)),
                                        )

    model.train()
    for epoch in range(1, args.epochs+1):
        print epoch
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)
            print 'feature'; print feature; print 'target'; print target
            optimizer.zero_grad()
            logit = model(feature)

        break



    results = {}; tag_to_idx = {"I": 0, "O": 1}
    for sentence, tags in testing_data:
        sentence_in = prepare_sequence(sentence, word_to_idx)
        scores = model(sentence_in)
        labels = [to_scalar(torch.max(item, 0)[1]) for item in tag_scores]
        true = [tag_to_idx[item] for item in tags];
        results[str(len(results))] = {'tokens':sentence, 'true':true, 'pred': labels}
                
    f = open('/nlp/data/romap/law/task_3/results.json', 'w+')
    f.write(json.dumps(results))

def get_args():
    parser = argparse.ArgumentParser(description='CNN')


    parser.add_argument('-model_name', type=str, default='cnn', help='dirname for model')
    parser.add_argument('-set_name', type=str, default='concept', help='dirname for wvecs')
    parser.add_argument('-input_channel',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-output_channel',  type=int, default=20,   help='how many steps to wait before logging training status [default: 1]')


    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # data 
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    global set_name, wvec_paths, model_name
    wvec_paths = {'google': '/nlp/data/corpora/GoogleNews-vectors-negative300.bin',
                                   'legal': '/nlp/data/romap/ambig/w2v/w2v100-300.txt',
                                   'concept': '/nlp/data/romap/conceptnet/numberbatch-en-17.06.txt'
                                   }
    #set_name = sys.argv[1]; wvec_path = wvec_paths[set_name]
    #model_name = sys.argv[2]
    train()
