from utils import loaddata , get_pred , displayconfusionmatrix ,tokenize,split_data,text2seq
from tqdm import tqdm
import nltk
from classifiers import getmodels

def get_pos_tags(data):
    tags =[]
    if type(data[0]) == str:
        data = tokenize(data)
    for x in tqdm(data):
        mapping = nltk.pos_tag(x)
        tags.append([x[1] for x in mapping])
    return tags


X, y = loaddata()
posdata =get_pos_tags(X)
X_train, X_test, y_train, y_test = split_data(X,y)
X_train , X_test = text2seq(X_train ,X_test)
models =getmodels()


models = getmodels()
report = get_pred(models,X_train, X_test, y_train, y_test )
displayconfusionmatrix(report[0] , 'POS')
displayconfusionmatrix(report[1] , 'POS')