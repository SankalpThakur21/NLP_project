import keras
import pickle
import tensorflow as tf
from nltk import word_tokenize
import numpy as np
model = keras.models.load_model("src/model.h5", compile=False)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5),loss='categorical_crossentropy',metrics =['accuracy'])

with open("src/word2index.pkl",'rb') as fp:
    word2index = pickle.load(fp)

def getseq(sent):
    res = []
    for token in word_tokenize(sent):
        if token in word2index:
            res.append(word2index[token])
        else:
            res.append(1)
    res = tf.keras.utils.pad_sequences([res] ,maxlen = 60, padding = 'pre')

    return res

print("0 : fluent   1 : neutral 2 : non-fluent")
sentence1 = "How are you?"
sentence2 = "I am fine"
sentence3 = "boats boats boats"
print(sentence1)
X = getseq(sentence1)
res = model.predict(X)
print("class :" ,np.argmax(res) , "prob :", res[0])

print(sentence2)
res = model.predict(X)
print("class :" ,np.argmax(res) , "prob :", res[0])

print(sentence3)
X = getseq(sentence3)

res = model.predict(X)
print("class :" ,np.argmax(res) , "  prob :", res[0])
