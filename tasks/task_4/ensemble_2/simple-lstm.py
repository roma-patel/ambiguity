import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import sys

embedding_dim, hidden_dim = 100, 100
#word_to_idx, tag_to_idx, training_data = {}, {}, []

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, word_to_idx, tagset_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(len(word_to_idx), embedding_dim)
        self.get_embeddings(word_to_idx, embedding_dim, wvec_path)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.wvec_path = wvec_path

        ## change hidden2tag
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def get_embeddings(self, word_to_idx, embedding_dim, wvec_path):
        print 'here'
        word_vecs = {}; vocab = word_to_idx.keys()
        if wvec_path[-3:] == 'bin':
            with open(fname, "rb") as f:
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

        elif name == 'txt':
            f = open(fname, 'r'); lines = f.readlines()[1:]
            for line in lines:
                items = lines.strip().split(' ')
                word, vec = items[0], [float(item) for item in items[1:]]
                if word in vocab: word_vecs[word] = vec
                
        
        weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
        mat = np.zeros((len(word_to_idx), embedding_dim))
        for word in word_to_idx:
            mat[word_to_idx[word]] = word_vecs[word]
        ######## add pretrained
        self.word_embeddings.weight.data.copy_(torch.from_numpy(mat))
        
        
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def to_scalar(var):
    return var.view(-1).data.tolist()[0]

def load_data():
    f = open('/nlp/data/romap/law/task_3/data/spec.json', 'r')
    for line in f: temp = json.loads(line)

    training_data = [(temp[key]['tokens'], temp[key]['tags']) for key in temp]
    return training_data


    
def prepare_data():
    #training_data = load_data()
    training_data = [
    ("It just depends on how we deploy the child".split(), ['I', 'O', 'O', 'I', 'O', 'O', 'I', 'O', 'I']),
    ("The term child means young adult .".split(), ['O', 'O', 'I', 'O', 'O', 'O', 'O']),
     ("The child was injured .".split(), ['O', 'I', 'O', 'O', 'O'])
    ]

    test_data = [
    ("The term child means person above 18 years .".split(),
             ["O", "O", "I", "O", "O", "O", "O", "O", "O"])
    ]
    word_to_idx = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    for sent, tags in test_data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
                
    print 'word_to_idx'
    print(word_to_idx)
    tag_to_idx = {"I": 0, "O": 1}
    return training_data, word_to_idx, tag_to_idx
    
def train():
    training_data, word_to_idx, tag_to_idx = prepare_data()
    #inputs = prepare_sequence(training_data[0][0], word_to_idx)

    print 'len word_to_idx, len tag_to_idx'
    print len(word_to_idx); print len(tag_to_idx)
    model = LSTM(embedding_dim, hidden_dim, len(word_to_idx), len(tag_to_idx))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(25):
        for sentence, tags in training_data:
            print sentence
            model.zero_grad()
            model.hidden = model.init_hidden()
            sentence_in = prepare_sequence(sentence, word_to_idx)
            targets = prepare_sequence(tags, tag_to_idx)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()


    test_data = [("The term child means person above 18 years .".split(),
             ["O", "O", "I", "O", "O", "O", "O", "O", "O"])]
    inputs = prepare_sequence(test_data[0][0], word_to_idx)
    tag_scores = model(inputs)
    print tag_scores
    labels = [to_scalar(torch.max(item, 0)[1]) for item in tag_scores]

    print labels
    
if __name__ == '__main__':
    global set_name, wvec_path
    wvec_paths = {'google': '/nlp/data/corpora/GoogleNews-vectors-negative300.bin',
                                   'legal': '/nlp/data/romap/ambig/w2v/w2v100-300.txt',
                                   'concept': '/nlp/data/romap/conceptnet/numberbatch-en-17.06.txt'
                                   }
    set_name = sys.argv[1]; wvec_path = wvec_paths[set_name]

    #testing
    print wvec_path
    training_data, word_to_idx, tag_to_idx = prepare_data()
    model = LSTM(embedding_dim, hidden_dim, word_to_idx, len(tag_to_idx))

    #train()
