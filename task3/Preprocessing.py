import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import preprocessing

def makePre(bonus = False):
    if bonus == False:
        df = pd.read_csv("penguins.csv")
        enc = LabelEncoder()
        df["gender"] = enc.fit_transform(np.array(df["gender"]).reshape(-1, 1))
        y = df["species"]
        x = df.drop(columns = "species")
        y = pd.get_dummies(y)
        xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 20)
        xtrain = (xtrain.iloc[:].values).astype('float32')
        ytrain = ytrain.iloc[:].values.astype('int32')
        xtest = (xtest.iloc[:].values).astype('float32')
        ytest = ytest.iloc[:].values.astype('int32')
        return xtrain,xtest,ytrain,ytest
    else :
        #read dataset
        train_data = pd.read_csv("data_set\\mnist_train.csv")
        test_data = pd.read_csv("data_set\\mnist_test.csv")
        X_train = (train_data.iloc[:,1:].values).astype('float32')
        y_train = train_data.iloc[:,0].values.astype('int32')
        X_test = (test_data.iloc[:,1:].values).astype('float32')
        y_test = test_data.iloc[:,0].values.astype('int32')
        #Convert to img
        X_train = X_train.reshape(X_train.shape[0], 28, 28)
        X_test = X_test.reshape(-1, 28, 28,1)
        for i in range(9):
            plt.subplot(330 + (i+1))
            plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
            plt.title(y_train[i])
        X_train = X_train.reshape(X_train.shape[0], 784)
        X_test = X_test.reshape(X_test.shape[0], 784)
        X_train = X_train.astype('float32')/255
        X_test = X_test.astype('float32')/255 
        new_y_train = np.zeros((y_train.shape[0],10))
        for i in range(len(y_train)):
            a = np.zeros((1,10))
            a[0,y_train[i]] =1
            new_y_train[i] = a
        new_y_test = np.zeros((y_test.shape[0],10))
        for i in range(len(y_test)):
            a = np.zeros((1,10))
            a[0,y_test[i]] =1
            new_y_test[i] = a
            #X_train = preprocessing.normalize(X_train)
            #X_test =  preprocessing.normalize(X_test)

        return X_train,X_test,new_y_train,new_y_test
