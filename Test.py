import numpy as np
import tensorflow as tf
import os
os.chdir('dir') 

test_batch_size = 5000
test_iters=150001

X_test=np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=0)
y_test=np.loadtxt(open("y_test.fasta","rb"),delimiter=",",skiprows=0)

def next_batch(train_data, train_target, batch_size):  
    index = [ i for i in range(0,76657) ] #number of testing sequences
    np.random.shuffle(index);  
    batch_data = []; 
    batch_target = [];  
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]);  
        batch_target.append(train_target[index[i]])  
    return batch_data, batch_target


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess=tf.InteractiveSession() 
    sess.run(init)
   
    model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess,model_file)
    
    step = 1
    # Keep training until reach max iterations
    while step * test_batch_size < test_iters:
        test_x, test_y= next_batch(X_test,y_test,test_batch_size)
        arr = np.array(test_x)
        embedding = np.loadtxt(open("embedding_matrix.csv","rb"),delimiter=",",skiprows=0)
        a=np.zeros(shape=(arr.shape[0],arr.shape[1],20)) #number of embedding size

        for i in range(0,arr.shape[0]):
            for j in range(0,arr.shape[1]):
                b=int(arr[i,j])
                a[i,j,:]=embedding[b]
        c=a.tolist()
       
        if step % display_step == 0 :
            # Calculate batch accuracy
            acc =sess.run(accuracy, feed_dict={x: c, y: test_y})
           
            print("Iter " + str(step * test_batch_size) + ", Minibatch ACC= " + "{:.6f}".format(acc))
    
            y_score=sess.run(pred,feed_dict={x:c})
            np.savetxt('score.csv',y_score, delimiter = ',')
 
            metrics.roc_auc_score(np.array(test_y).ravel(), np.array(y_score).ravel(), average='micro')
            fpr, tpr, thresholds = metrics.roc_curve(np.array(test_y).ravel(),np.array(y_score).ravel())
            auc = metrics.auc(fpr, tpr)
            print("AUC= "+ "{:.6f}".format(auc))
    
        step += 1
    sess.close()  

np.savetxt('score.csv',y_score, delimiter = ',')
