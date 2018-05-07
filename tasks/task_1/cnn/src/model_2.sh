source /home1/r/romap/neuralenv/nn/bin/activate
#python /nlp/data/romap/law/task_1/cnn/src/process_data.py /nlp/data/corpora/GoogleNews-vectors-negative300.bin
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python /nlp/data/romap/law/task_1/cnn/src/conv_net_sentence.py -nonstatic -word2vec