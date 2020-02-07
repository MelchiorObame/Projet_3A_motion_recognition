#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:43:16 2020

@author: saadbenda
"""
import argparse
parser = argparse.ArgumentParser(description='rnn')
parser.add_argument('--test', help='train or test',action='store_true')
parser.add_argument('--train', help='train or test',action='store_true')
parser.add_argument('--db', help='action,emotion,dance')
parser.add_argument('--type', help='lstm,cnn+lstm,convlstm')
args = parser.parse_args()


#import numpy as np 
#
#from sklearn.preprocessing import MinMaxScaler
#
#
#from tensorflow import keras
#
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import normalize
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt 
import tensorflow as tf
#import keras
from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping,CSVLogger,ReduceLROnPlateau, ModelCheckpoint
#from keras.callbacks import EarlyStopping,CSVLogger,ReduceLROnPlateau, ModelCheckpoint
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.models import Sequential
#from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
#from sklearn.model_selection import train_test_split
#from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import backend

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
print(tf.__version__)


def getData():
    
    if args.db=='action':
        df=np.loadtxt(r'Action3dData.csv',dtype=str,delimiter=';',skiprows=1,usecols=(1,2,3,8))
        i=0
        k=0
        labels=[]
        
        datanp=[]
        l=[]
        nb_frame=int(len(df)/20)

        for i in range(nb_frame):
                d=df[k:k+20,:-1]
                d=d.ravel()
                l.append(d)

                labels.append(df[k,-1])
                k=k+20
        l=np.asarray(l)
        labels=np.asarray(labels)        
        num_class=np.unique(labels)
        print('labels',num_class)
        #print('data',l[:3,:])
        print('labels',type(labels))
        print('labels_shape',labels.shape)
        print('data_shape',l.shape)

    if args.db=='dance':
        df=np.loadtxt(r'DanceData.csv',dtype=str,delimiter=';',skiprows=1,usecols=(1,2,3,4,5,6,9))
        i=0
        k=0
        labels=[]
        datanp=[]
        l=[]
        nb_frame=int(len(df)/54)
        for i in range(nb_frame):
                d=df[k:k+54,:-1]
                d=d.ravel()
                l.append(d)
                labels.append(df[k,-1])
                k=k+54
        l=np.asarray(l)
        labels=np.asarray(labels)        
        num_class=np.unique(labels)
    
    if args.db=='emotion':
        df=np.loadtxt(r'EmotionalData.csv',dtype=str,delimiter=';',skiprows=1,usecols=(1,2,3,4,5,6,9))
        i=0
        k=0
        labels=[]
        
        datanp=[]
        l=[]
        nb_frame=int(len(df)/23)
        nb_frame=46
        for i in range(nb_frame):
                #print('progression : %s/%s\n'%(i,lendf))
                
                d=df[k:k+23,:-1]
                #d=np.array()
                #print('dshape',d.shape)
                d=d.ravel()
                #print('sq',d)
                #print(type(d))
                #print(d.shape)
                l.append(d)
                
                #print('newdata',type(data))
                #print('newdata',data.shape)
                
                #d=np.array(d).ravel()
                #np.array(data.append(np.array(d).ravel()))
                
                labels.append(df[k,-1])
                k=k+23
        l=np.asarray(l)
        labels=np.asarray(labels)        
        num_class=np.unique(labels)
    return l,labels,num_class
    
#    with open ('data.csv','w+') as txt:
#            with np.printoptions(threshold=np.inf):
#                headerLine = "x\n"
#                txt.write(headerLine)
#                #print('skes_joints',skes_joints[5,3])
#                print(c.c,file=txt)
    
def scale(train, test):
    # fit
    #print('train',train.shape)
    np.asarray(train)
    np.asarray(test)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled    


"""Transformer les classes en représentation binaire (one-hot encoding)"""
def transform_labels(y_train,y_test):
  """
  Cette fonction transform les classes non-binaire à une représentation binaire
  Par exemple si on a une liste de 6 fleures chacune peut avoir une des 3 classes
  """
  print('y_train',y_train) 
  print('y_test',y_test) 
  
  # concatener train et test  
  y_train_test = np.concatenate((y_train,y_test),axis =0)
  
  # init un encoder Label 
  encoder = LabelEncoder()
  # transformer de [1,3,3,2,1,2] à [0,2,2,1,0,1] 
  new_y_train_test = encoder.fit_transform(y_train_test)
  
  # init un encoder one-hot 
  encoder = OneHotEncoder()
  # transformer de [0,2,2,1,0,1] à la représentation binaire
  new_y_train_test = encoder.fit_transform(new_y_train_test.reshape(-1,1))
  #print('new_y_train_test',new_y_train_test)
  
  # resplit the train and test
  new_y_train = new_y_train_test[0:len(y_train)]
  new_y_test = new_y_train_test[len(y_train):]
def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 60))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector

  
  


def one_hot_vector(labels,num_class):
    # init un encoder Label 
    encoder = LabelEncoder()
    # transformer de [1,3,3,2,1,2] à [0,2,2,1,0,1] 
    new_labels = encoder.fit_transform(labels)
    
    
    # init un encoder one-hot 
    #encoder = OneHotEncoder()
    # transformer de [0,2,2,1,0,1] à la représentation binaire
    print('num_class',len(num_class))
    num_skes = len(new_labels)
    labels_vector = np.zeros((num_skes, len(num_class)))
    print('ioqdsdhq-----',labels_vector.shape)
    
    for idx, l in enumerate(new_labels):
        labels_vector[idx, l] = 1
    
    
    #new_labels = encoder.fit_transform(new_labels.reshape(-1,1))
    #print('new_y_train_test',new_y_train_test)
    
    
    print('new_labels',labels_vector.shape)
    return labels_vector
def split_train_test(percentage,data):
    split=int(percentage*data.shape[0])
    train=data[:split,:]   
    test=data[split:,:]
    
    return train,test


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
def epoch_loss_acc(mode,history):
    '''variation du taux d'erreur sur le train et sur le validation set en fonction du nombre d'epoque'''
    history_dict = history.history
    loss_train_epochs = history_dict[mode]
    loss_val_epochs = history_dict['val_'+mode]
    
    plt.figure()
    plt.plot(loss_train_epochs,color='blue',label='train_'+mode)
    plt.plot(loss_val_epochs,color='red',label='val_'+mode)
    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.legend()
    plt.savefig('epoch-'+mode+'.pdf')
    plt.show()
    plt.close()
    
def evaluate_model(x_train,y_train,x_test,y_test,nnType,batch_size=1,epochs=60):
    backend.clear_session()
    n_timesteps, n_features, n_outputs = x_train.shape[0], x_train.shape[1], y_train.shape[1]
    nb_neurons_lstm=100
    if args.type=="cnn+lstm":
        
        # reshape data into time steps of sub-sequences
        #[batch, timesteps, feature].
        
        epochs=25
        n_steps, n_length = 4, 32
        
        
        # define model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(units=nb_neurons_lstm))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        x_train_reshape=x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))
        x_test_reshape=x_train.reshape((x_test.shape[0], n_steps, n_length, n_features))
    
    elif args.type=="lstm":
        
        '''64 windows of data will be exposed to the model before the weights of the model are updated.'''
        
        nb_classes=y_train.shape[1]
        print('nb_classes',nb_classes)
        
        
        model = Sequential()
        model.add(LSTM(units=nb_neurons_lstm, return_sequences=True,input_shape=(1,n_features)))
        model.add(LSTM(units=nb_neurons_lstm, return_sequences=True))
        model.add(LSTM(units=nb_neurons_lstm))
        
        
        #model.add(LSTM(units=nb_neurons_lstm))
        '''This is followed by a dropout layer intended to reduce overfitting of the model to the training data.'''
        model.add(Dropout(0.5))
        '''Activation function is softmax for multi-class classification.'''
        #model.add(Dense(100, activation='relu'))
        model.add(Dense(units=n_outputs, activation='softmax'))
        model.summary()
        '''Because it is a multi-class classification problem, categorical_crossentropy is used as the loss function.  '''
        # reshape pour avoir un shape: (sample, timestamps, features)
        
        print('x_train-------------',x_train.shape[0])
        x_train_reshape = x_train.reshape(x_train.shape[0],1,x_train.shape[1])#60 features
        x_test_reshape = x_test.reshape(x_test.shape[0],1,x_test.shape[1])#60 features
        
    elif args.type=="convlstm":
        n_steps, n_length = 4, 32
        # define model
        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.summary()

        # reshape into subsequences (samples, time steps, rows, cols, channels)
        x_train_reshape = x_train.reshape((x_train.shape[0], n_steps, 1, n_length, n_features))
        x_test_reshape = x_test.reshape((x_test.shape[0], n_steps, 1, n_length, n_features))

    filepath = 'modelcheckpoint_'+str(args.db) + '.hdf5'
    saveto = 'csvLogger_'+str(args.db) + '.csv'
    #optimizer = Adam(lr=lr, clipnorm=args.clip)
    #pred_dir = os.path.join(rootdir, str(case) + '_pred.txt')
    """spécifier d'utiliser une partie du train pour la validation des hyper-paramèteres"""
    percentage_of_train_as_validation = 0.3
    if args.train:
        early_stop = EarlyStopping(monitor='val_accuracy', patience=15, mode='auto',min_delta=0.0001)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, mode='auto', cooldown=3., verbose=1)
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
        csv_logger = CSVLogger(saveto)
        callbacks_list = [csv_logger, checkpoint, early_stop, reduce_lr]
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        '''Here we do shuffle the windows of input data during training (the default). In this problem, we are interested in harnessing the LSTMs ability to learn and extract features across the time steps in a window, not across windows.'''
        
        history = model.fit(x_train_reshape, y_train, epochs=20,verbose=True, batch_size=10,validation_split=percentage_of_train_as_validation,callbacks=callbacks_list)
        #model.fit(train_x, train_y, validation_data=[valid_x, valid_y], epochs=args.epochs,batch_size=args.batch_size, callbacks=callbacks_list, verbose=2)
        
        
        epoch_loss_acc('accuracy',history)#loss or accuracy
        epoch_loss_acc('loss',history)#loss or accuracy
        
        loss,accuracy = model.evaluate(x_train_reshape,y_train,verbose=False)
        print("L'accuracy sur l'ensemble du train est:",accuracy)
        print("Le loss sur l'ensemble du train est:",loss)
    elif args.test:
        model.load_weights(filepath)
        #model = keras.models.load_model('best-model.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # evaluate model
        loss,accuracy = model.evaluate(x_test_reshape, y_test, batch_size=batch_size, verbose=False)
        print("L'accuracy sur l'ensemble du test est:",accuracy)
        print("Le loss sur l'ensemble du train est:",loss)
        
        #scores = get_activation(model, test_x, test_y, pred_dir, VA=10, par=9)
        #results.append(round(scores, 2))
    return accuracy
 
def main(repeats=1):
    nnType=args.type
    
    
    df,labels,num_class=getData()
 
    train,test=split_train_test(0.6,df)

    scaler, train_scaled, test_scaled = scale(train, test)
    
    print('train_scaled_shape',train_scaled.shape)
    #print('train_scaled',train_scaled[:2,:])
    print('test_scaled_shape',test_scaled.shape)
    
    x_train=train_scaled
    x_test=test_scaled
    
    '''y maintenant'''
    new_labels=one_hot_vector(labels,num_class)
    y_train,y_test=split_train_test(0.6,new_labels)
    
    print('---------------------------------------------------')
    print('x_train',x_train.shape)
    print('x_test',x_test.shape)
    print('y_train',y_train.shape)
    print('y_test',y_test.shape)
    #accuracy=evaluate_model()
    
    # repeat experiment
    scores = list()
    '''We cannot judge the skill of the model from a single evaluation.
The reason for this is that neural networks are stochastic, meaning that a different specific model will have the same result when training the same model configuration on the same data.'''
    for r in range(repeats):
        score = evaluate_model(x_train, y_train, x_test, y_test,nnType)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)
    
