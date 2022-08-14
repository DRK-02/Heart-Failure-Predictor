import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

model = 0
mms = 0

def db_create(dict):
    db = pd.DataFrame(dict)
    return db

def train():
    global model, mms

    db = 'https://raw.githubusercontent.com/DRK-02/Heart-Failure-Predictor/main/Data_Set.csv?token=GHSAT0AAAAAABXLXMDVTLVUJ5XN4H2YDKTGYXSIUCQ'
    df = pd.read_csv(db)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])

    df = df[df['RestingBP'] != 0]

    xdf = df.drop("HeartFailure", axis = 1)
    ydf = df["HeartFailure"]
    #x_train,x_test,y_train,y_test=train_test_split(xdf,ydf,test_size=0.01,random_state=20)

    mms = MinMaxScaler(feature_range = (0, 1))
    xdf=mms.fit_transform(xdf)
    #x_test=mms.fit_transform(x_test)
    xdf=pd.DataFrame(xdf)
    #x_test=pd.DataFrame(x_test)

    model = SVC(probability = True)
    model.fit(xdf, ydf)
'''
    predictions=model.predict(x_test)
    TP, FN, FP, TN = confusion_matrix(y_test,predictions,labels=[1,0]).reshape(-1)
    acy = round((TP+TN)/(TP+FP+TN+FN), 3)
    print(y_test, predictions, sep = '\n')
'''
def predictor(data):
    db = 'https://raw.githubusercontent.com/DRK-02/Heart-Failure-Predictor/main/Data_Set.csv?token=GHSAT0AAAAAABXLXMDVTLVUJ5XN4H2YDKTGYXSIUCQ'
    df = pd.read_csv(db)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])

    df = df[df['RestingBP'] != 0]

    xdf = df.drop("HeartFailure", axis = 1)
    ydf = df["HeartFailure"]
    x_train,x_test,y_train,y_test=train_test_split(xdf,ydf,test_size=0.005)
    x_test = pd.concat([x_test, data], axis = 0)

    mms = MinMaxScaler(feature_range = (0, 1))
    print(x_test)
    x_train=mms.fit_transform(x_train)
    x_train=pd.DataFrame(x_train)
    x_test=mms.fit_transform(x_test)
    x_test=pd.DataFrame(x_test)
    print(x_test)

    model = SVC(probability = True)
    model.fit(x_train, y_train)
    '''
    print(data)
    data = mms.fit_transform(data)
    data = pd.DataFrame(data)
    print(data)'''
    prediction = model.predict(x_test)
    print(prediction)

info = {
    'Age': [46],
    'Sex': [1],
    'ChestPainType': [1],
    'RestingBP': [140],
    'Cholesterol': [275],
    'FastingBS': [0],
    'RestingECG': [1],
    'MaxHR': [165],
    'ExerciseAngina': [1],
    'Oldpeak': [0],
    'ST_Slope': [2]
}


info = db_create(info)
predictor(info)
