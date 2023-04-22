

from keras import Sequential
from keras.layers import LSTM,Embedding, Dense, Input
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from nltk import word_tokenize, sent_tokenize
from utils import split_data,get_pred,loaddata , displayconfusionmatrix


embed = None
glv_size = 50
MAX_LEN = 60
def getglove(path ):
    glv = dict()
    with open(path,'r' ,encoding="utf8") as fp:
        for line in fp:
            word, *vec = line.split()
            glv[word] = vec
    return glv
        

def makemodel( ):
    model = Sequential()
    emb = Embedding(embed.shape[0] , glv_size , weights =[embed],input_length =MAX_LEN ,trainable = False)
    model.add(emb)
    model.add(LSTM(256,return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(1000))
    model.add(Dense(3 , activation = 'softmax'))
    
    model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics =['accuracy'])
    model.summary()
    
    return model

def preparedata(X ,y ,glv , glv_size , MAX_LEN = 60):
    global embed
    embed = np.zeros((len(glv)+2 , glv_size))
    ind =2
    word2index ={'_pad_':0}
    index2word ={0:'_pad_'}

    for x in glv:
        if(len(glv[x]) != glv_size):
            continue
        embed[ind] = glv[x]
        word2index[x] = ind
        index2word[ind] = x
        ind+=1

    print("--creating-tokens-----")
    tokens_x  =[]
    for sentences in X:
        tokens_x.append(word_tokenize(sentences.lower()))

    print("-----text2seq------")
    seq_X = []
    for sentence_tokens in  tokens_x:
        cur = []
        for token in sentence_tokens:
            if token in word2index:
                cur.append(word2index[token])
            else:
                cur.append(1)
        seq_X.append(cur)
        
    trainX , testX , trainy , testy = split_data(seq_X,y)

    print("----padding---")

    trainX = tf.keras.utils.pad_sequences(trainX ,maxlen = MAX_LEN, padding = 'pre')
    testX = tf.keras.utils.pad_sequences(testX ,maxlen = MAX_LEN, padding = 'pre')
    
    return trainX ,testX, trainy ,testy


path = "data/glove.6B.{}d.txt".format(glv_size)
X,y = loaddata()
glv = getglove(path)
trainX ,testX, trainy ,testy = preparedata(X ,y ,glv , glv_size , MAX_LEN)

report = get_pred({"LSTM-glove" : makemodel} , trainX ,  testX ,trainy , testy , nn=1)

displayconfusionmatrix(report[0] , 'Glove-lstm')
displayconfusionmatrix(report[1] , 'Glove-lstm')

