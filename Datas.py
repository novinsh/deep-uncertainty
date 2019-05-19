#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/bin/python
import numpy as np
import keras.utils
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
train = unpickle('test_batch')


def one_hot_representation(dlabel):
    col = len(dlabel)
    one_hot = np.zeros((10,col))
    k = 0
    for i in dlabel:
        one_hot[i][k] = 1
        k = k + 1
    return one_hot
def LoadBatch(train):
    trX = np.transpose(train[b'data'])
    trX = np.array(trX,dtype = float)
    trX = trX /255
    trY = keras.utils.to_categorical(train[b'labels'], num_classes=10)
  
    mean_x = np.mean(trX , axis=0)
    std_x = np.std(trX , axis=0)
    std_x = np.array(std_x)
    trX = (trX - mean_x)
    trX = trX/std_x
    trsy = train[b'labels']
    return trX,trY,trsy
'''to try to build the whole dataset 
    full_or_not : 
            1 if we want to use the whole batch data, 
            0 if we only take one batch as training and one batch as test'''
def Initialize_data(full_or_not):
    if full_or_not is 0 :
        # TRAINING
        train = unpickle('data_batch_1')
        (trx,trY,trsy) = LoadBatch(train)
        # VALIDATION
        val   = unpickle('data_batch_2')
        (valx,valY,valy) = LoadBatch(val)
        # TEST
        test  = unpickle('test_batch')
        (testx,testY,testy) = LoadBatch(test)
    else: 
        # TRAINING , val
        
        for i in range(5):
            name = 'data_batch_' + str(i+1)
            temp = unpickle(name)
            (tempx,tempY,tempy) = LoadBatch(temp)
            if i == 0 :
                trxt  = tempx
                trYt  = tempY
                trsyt = tempy
            else:
                trxt = np.concatenate((trxt,tempx),axis=1)
                trYt = np.concatenate((trYt,tempY),axis=0)
                trsyt = np.concatenate((trsyt,tempy),axis = 0)
              
        trx = trxt[:,0:48999]
        #print(np.shape(trx)) 
        valx = trxt[:,49000:49999]
        #print(np.shape(valx)) 
        trY = trYt[0:48999][:]
        valY = trYt[49000:49999][:]
        trsy = trsyt[0:48999]
        valy = trsyt[49000:49999]
        
        # test
        
        test  = unpickle('test_batch')
        (testx,testY,testy) = LoadBatch(test)
    return (trx.T,trY,trsy,valx.T,valY,valy,testx.T,testY,testy)
      
def category_names(dataset):
    if dataset == 'cifar10':
        return [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck']
    else:
        raise ValueError("Unexpected dataset " + dataset + ".")

def ComputeAccuracy(predictions,testY):
    answers = predictions.argmax(axis=1)
    (d, ) = np.shape(testY)
    ii = np.reshape(np.array(testY),(d,1))
    y = np.reshape(np.array(answers),(d,1))		
    correct = ii - y
    correct[correct!=0] = -1
    correct[correct==0] = 1
    correct[correct==-1] = 0
    a = np.count_nonzero(correct)
    result = float(a)/float(d)
    return result

def Draw_loss(epochs,predictions,H):
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    #### print accuracy
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title(" Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    #### print loss
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss ")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
   