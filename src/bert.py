
import torch
from tqdm import tqdm
from transformers import BertTokenizer ,BertModel
from utils import split_data , loaddata ,get_pred , displayconfusionmatrix
from keras import Sequential
from keras.layers import LSTM,Embedding, Dense, Input

def loadbert():
    with torch.no_grad():
            bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    bm = BertModel.from_pretrained('bert-base-uncased' , output_hidden_states =True)
    return bm, bert_tokenizer

def encode(data ,bm , bert_tokenizer):
    output = []
    for sentence in tqdm(data):
        z = ["[CLS]"] + bert_tokenizer.tokenize(sentence.lower()) + ["[SEP]"]
        tensor_inp = torch.Tensor([bert_tokenizer.convert_tokens_to_ids(z)]).long()
        seg = torch.ones((1,len(z))).long()
        with torch.no_grad():
            output.append(torch.mean(bm(tensor_inp,seg).hidden_states[0][0],axis = 0))
        
    return torch.stack(output)


def makemodel():
    model = Sequential()
    model.add(Input(shape = (768,) ))
    model.add(Dense(768))
    model.add(Dense(1000))
    model.add(Dense(1000))
    model.add(Dense(3 , activation = 'softmax'))
    model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics =['accuracy'])
    model.summary()
    
    return model

X,y = loaddata()
bm,bert_tokenizer = loadbert()
encodings = encode(X , bm ,bert_tokenizer)
trainx , testx , trainy , testy = split_data(encodings , y)

report = get_pred({"bert":makemodel} , trainx.numpy() ,testx.numpy() ,  trainy , testy , nn =1)
displayconfusionmatrix(report[0] , 'Bert')
displayconfusionmatrix(report[1] , 'Bert')
