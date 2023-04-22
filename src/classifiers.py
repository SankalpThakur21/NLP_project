from utils import loaddata , build_data , get_pred , displayconfusionmatrix
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



def getmodels():
    LR = LogisticRegression(random_state=9098,max_iter = 250)
    SVM = make_pipeline(StandardScaler(), SVC(random_state = 100,max_iter = 2500,gamma='auto'))
    RF = RandomForestClassifier()
    XGB  = XGBClassifier()

    models = {'logisticRegression' : LR , 'SVM' :SVM , "RandomForest":RF  ,"XGB":XGB}
    return models

X, y = loaddata()
X_train, X_test, y_train, y_test = build_data(X,y)

models = getmodels()
report = get_pred(models,X_train, X_test, y_train, y_test )
displayconfusionmatrix(report[0] , 'classifiers')
displayconfusionmatrix(report[1] , 'classifiers')
