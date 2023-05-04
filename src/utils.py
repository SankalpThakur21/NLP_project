import os
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import LSTM,Embedding, Dense, Input
from keras.preprocessing.text import Tokenizer

import pandas as pd
import numpy as np
import nltk
from nltk.util import pad_sequence
from nltk import word_tokenize, sent_tokenize

from imblearn.over_sampling import RandomOverSampler, SMOTE ,BorderlineSMOTE,ADASYN

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score,  classification_report



samplers = {"Nosampling" : None , "SMOTE" : SMOTE,  "BorderlineSMOTE" : BorderlineSMOTE,"ADASYN" : ADASYN}

def remove_unwanted(mt , ref , avgm , avgg):
  src = []
  trans = []
  fl1 = []
  fl2 = []
  for i in range(len(ref)):
    ref_tok = word_tokenize(ref[i].lower())
    sent_tok = word_tokenize(mt[i].lower())
    if len(ref_tok) < 5:
      continue
    if len(sent_tok) < 5:
      continue
    src.append(ref_tok)
    trans.append(sent_tok)
    fl1.append(avgm[i])
    fl2.append(avgg[i])
  return src , trans , fl1 ,fl2
    

def loaddata(path = "data/data.tsv"):
    # read the data
    data  =pd.read_table(path,usecols=['Source' , 'Shortening' , 'AverageMeaning','AverageGrammar'])

    # created classes
    cdata = data[data['AverageMeaning']<=3 ][data['AverageGrammar']<=3]
    cdata['fluency'] = 0.2 * cdata['AverageMeaning'] + 0.8 * cdata['AverageGrammar']
    cdata['fluency'] = cdata['fluency'].round()

    X = cdata['Shortening']
    y = np.int32(cdata['fluency'])-1
    
    return X,y


def split_data(X,y):
    return train_test_split(X, y, test_size=0.33, random_state=42)


def pad_sequences(X , Xtest):
    MAX_LEN=60
    for x in X:
        MAX_LEN = max(MAX_LEN , len(x))
    X = tf.keras.utils.pad_sequences(X ,maxlen = MAX_LEN, padding = 'pre')
    Xtest = tf.keras.utils.pad_sequences(Xtest ,maxlen = MAX_LEN, padding = 'pre')
    return X , Xtest


def tokenize(text):
    tokens = []
    for data in text:
        sentences = sent_tokenize(data)
        cursent = []
        for sentence in sentences:
            cursent.extend(word_tokenize(sentence.lower()))
        tokens.append(cursent)
    return tokens


def text2seq(X_train , X_test):
    wordtoken = Tokenizer(oov_token = 1)
    tokenized_data = []
    tokenized_X  = tokenize(X_train)
    tokenized_testX  = tokenize(X_test)
    
    
    wordtoken.fit_on_texts(tokenized_X)
    X = wordtoken.texts_to_sequences(tokenized_X)
    Xtest = wordtoken.texts_to_sequences(tokenized_testX)
    
    Xtrain , Xtest = pad_sequences(X,Xtest)
    
    return Xtrain , Xtest


def build_data(X,y):
    X_train, X_test, y_train, y_test = split_data(X,y)
    
    X_train = list(X_train)
    y_train = list(y_train)
    X_test = list(X_test)
    y_test = list(y_test)
    
    Xtrain , Xtest = text2seq(X_train , X_test)
    
    return Xtrain,Xtest , y_train , y_test 


def get_pred(models , X,Xtest ,y_train,y_test, samplers = samplers , nn=0):
    results_train = dict()
    results_test = dict()
    for sampler_name in samplers:
        print("-"*20)
        x = samplers[sampler_name]
        if(x!=None):
            print("trying {} resampler".format(sampler_name))
            sampler = x()
            cX,cy = sampler.fit_resample(X,y_train)
        else:
            cX , cy = X.copy() ,y_train.copy() 

        for name in models:
            model = models[name]
            if nn == False:
                model.fit(cX , cy)
                trainpred = model.predict(cX)
                testpred = model.predict(Xtest)
            else:
                model1 = model()
                trainy_cat = to_categorical(cy)
                testy_cat = to_categorical(y_test)
                callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
                model1.fit(cX , trainy_cat , validation_data = (Xtest , testy_cat),batch_size = 128,epochs = 25 , callbacks=[callback])
                trainpred = np.argmax(model1.predict(cX) , axis = 1)
                testpred = np.argmax(model1.predict(Xtest) , axis = 1)
            print(name)
            print("train accuracy : ", accuracy_score(trainpred,cy))
            print("test accuracy : ", accuracy_score(testpred , y_test))

            results_train[sampler_name + '_' + name +'_train'] =  [trainpred , cy]
            results_test[sampler_name + '_' + name +'_test'] = [testpred , y_test]
    return results_train, results_test

def get_report(res):
    reports = dict()
    for  x in res:
        reports[x] = classification_report(res[x][1] , res[x][0])
    return reports


def displayconfusionmatrix(res ,fol):
    for x in res:
        z = tuple(x.split("_"))
        ConfusionMatrixDisplay.from_predictions(res[x][1] ,res[x][0])
        results_dir = "Results/{}/{}".format(fol ,z[0])
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        rep  = classification_report(res[x][1] , res[x][0])
        with open("{}/{}_{}.json".format(results_dir , z[1] , z[2]), 'w' ) as fp:
            json.dump(rep , fp ,indent =3)
        plt.savefig("{}/{}_{}.png".format(results_dir , z[1] , z[2]),dpi =300)