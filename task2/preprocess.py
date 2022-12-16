import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import addline_model
from sklearn.preprocessing import MinMaxScaler
import  numpy as np
import seaborn as sys
#drow line
def Line(x_test,f1,f2,bias,c1_x_test,c2_x_test,weight):
    min1= min(x_test[f1])
    max1 = max(x_test[f1])
    x = [(min1 - 1), (max1 + 1)]
    if (bias == 1):
        y = - (weight[2] + np.multiply(weight[0], x)) / weight[1]
    else:
       y= - (np.multiply(weight[0], x)) / weight[1]
    plt.figure(figsize=(8, 6))
    plt.scatter(c1_x_test[f1], c1_x_test[f2], marker='+', color='green')
    plt.scatter(c2_x_test[f1], c2_x_test[f2], marker='_', color='red')

    plt.plot(x, y, label='Decision Boundary')
    plt.show()


#show matrix
def matrix(matrix,class1,class2):

    Matrix = np.array([[matrix[0], matrix[3]], [matrix[2], matrix[1]]])
    plt.figure(figsize=(8, 6))
    x = sys.heatmap(Matrix, xticklabels=[class1,class2], yticklabels=[class1,class2], annot=True)
    plt.show()

#run
def fire(learning_rate, feature1, feature2, class1, class2, epochs,mse, bais=0):
    data = pd.read_csv("penguins.csv", usecols=['species', feature1, feature2])
    data = data[(data['species'] == class1) | (data['species'] == class2)]

    ##encoding and preprocessing
    labelEnconer = LabelEncoder()

    if (feature1 == "gender" or feature2 == "gender"):
        data.fillna('male', inplace=True)
        data['gender'] = labelEnconer.fit_transform(data['gender'])


    # normalize data
    scaler = MinMaxScaler()
    normalize_data = data.drop('species', axis=1, inplace=False)
    normalize = pd.DataFrame(scaler.fit_transform(normalize_data), columns=normalize_data.columns)
    data[feature1] = normalize[feature1].values
    data[feature2] = normalize[feature2].values

    ##class1
    C1 = data[(data['species'] == class1)]
    C1 = shuffle(C1)
    C1_y = C1['species']
    C1_x = C1.drop('species', axis=1, inplace=False)
    C1_x_train, C1_x_test, C1_y_train, C1_y_test = train_test_split(C1_x, C1_y, test_size=0.4, shuffle=True,
                                                                    random_state=7)

    ##class2
    C2 = data[data['species'] == class2]
    C2 = shuffle(C2)
    C2_y = C2['species']
    C2_x = C2.drop('species', axis=1, inplace=False)
    C2_x_train, C2_x_test, C2_y_train, C2_y_test = train_test_split(C2_x, C2_y, test_size=0.4, shuffle=True,
                                                                    random_state=7)

    ##final data
    x_train = pd.concat([C1_x_train, C2_x_train])
    y_train = pd.concat([C1_y_train, C2_y_train])
    x_test = pd.concat([C1_x_test, C2_x_test])
    y_test = pd.concat([C1_y_test, C2_y_test])

    #incoding
    y_test = labelEnconer.fit_transform(y_test)
    y_train= labelEnconer.fit_transform(y_train)
    y_test=pd.DataFrame(y_test)
    y_test.replace(0,-1,True)
    y_train = pd.DataFrame(y_train)
    y_train.replace(0, -1, True)



    #run
    mod=addline_model.Model(itration=epochs, learning_rate=learning_rate, withBaise=bais, mse=mse)
    mod.fit(X=x_train.to_numpy(),Y=y_train.to_numpy())
    y_prd= mod.predict(X=x_test.to_numpy())

    # line
    weight = mod.weight()
    Line(x_test, feature1, feature2, bais, C1_x_test, C2_x_test, weight)
    # conf_matrix
    conf_matrix = mod.matrix_con(y_prd, y_test.to_numpy())
    matrix(conf_matrix, class1, class2)

    return mod.get_accuracy(Y_actual=y_test.to_numpy(),Y_prediction=y_prd)

