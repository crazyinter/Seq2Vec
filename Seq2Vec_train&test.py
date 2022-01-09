
# coding: utf-8

# In[ ]:


import os
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
import ast
import numpy as np
os.chdir('dir') # direct

# construct training dataset
m=[]
g=open('train_data','r')
lines=g.readlines()
for line in lines:
    n=line[:-1]
    k = ast.literal_eval(n)
    j=list(k)
    m.append(j) 
g.close()
train_x = m

# construct testing dataset
a=[]
f=open('test_data','r')
lines=f.readlines()
for line in lines:
    b=line[:-1]
    c = ast.literal_eval(b)
    d=list(c)
    a.append(d)
f.close()
test_x = a

# construct testing labels
test_y=[]
for i in range(0,500):
    test_y.append(1)
for i in range(500,1000):
    test_y.append(0)

# construct training labels
train_y=[]
for i in range(0,4500):
    train_y.append(1)
for i in range(4500,9000):
    train_y.append(0)
    
# vocabulary length
VOCAB_LEN = 65
# max sequence length
SEQUENCE_LEN = 498
# padding
train_x = pad_sequences(train_x, maxlen=SEQUENCE_LEN, value=0.)
test_x = pad_sequences(test_x, maxlen=SEQUENCE_LEN, value=0.)
train_y = to_categorical(train_y, 2)
test_y = to_categorical(test_y, 2)

# embedding size
WORD_FEATURE_DIM = 20

DOC_FEATURE_DIM = 64
net = tflearn.input_data([None, SEQUENCE_LEN])
net = tflearn.embedding(net, input_dim=VOCAB_LEN, output_dim=WORD_FEATURE_DIM)
net = tflearn.lstm(net, DOC_FEATURE_DIM, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0002,loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0,tensorboard_dir="/tmp/tflearn_logs/")

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' 
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# training
model.fit(train_x, train_y, validation_set=(test_x, test_y), show_metric=True,n_epoch=25, batch_size=256)

