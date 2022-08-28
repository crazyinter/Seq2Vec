import os
os.chdir('dir') 
import numpy as np
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics


X_train=np.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=0)
y_train=np.loadtxt(open("y_train.csv","rb"),delimiter=",",skiprows=0)


def next_batch(train_data, train_target, batch_size):  
    index = [ i for i in range(0,220151) ] #number of training sequences
    np.random.shuffle(index);  
    batch_data = []; 
    batch_target = [];  
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]);  
        batch_target.append(train_target[index[i]])  
    return batch_data, batch_target

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    
    inputs = tf.concat(inputs, 2)
    #hidden_size = n_hidden
    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    
    if not return_alphas:
        return output
    else:
        return output, alphas

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.reset_default_graph()

# Parameters
learning_rate = 0.000022
training_iters = 300000
batch_size = 128
display_step = 10
seq_max_len = 498 
n_hidden = 498 # hidden layer num of features
n_classes = 2 
ATTENTION_SIZE = 20

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len,20])
y = tf.placeholder("float", [None, n_classes])
seqlen=np.array([6,]) #序列长度

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32,sequence_length=seqlen)
    outputs=attention(outputs, attention_size=ATTENTION_SIZE, time_major=False, return_alphas=False)
    value = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)

    return tf.matmul(last, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess: 
    sess=tf.InteractiveSession() 
    sess.run(init)

    is_train=True
    saver=tf.train.Saver(max_to_keep=1)

    #训练阶段
    if is_train:
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y= next_batch(X_train,y_train,batch_size)
            arr1 = np.array(batch_x)
            embedding = np.loadtxt(open("embedding_matrix.csv","rb"),delimiter=",",skiprows=0)
            m=np.zeros(shape=(arr1.shape[0],arr1.shape[1],20)) #number of embedding size

            for i in range(0,arr1.shape[0]):
                for j in range(0,arr1.shape[1]):
                    n=int(arr1[i,j])
                    m[i,j,:]=embedding[n]
            
            k=m.tolist()
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: k, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: k, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: k, y: batch_y})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1

            saver.save(sess,'ckpt/train.ckpt',global_step=step+1)
        print("Optimization Finished!")
    sess.close()   


