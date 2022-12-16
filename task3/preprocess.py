import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import  numpy as np
import seaborn as sys
import mlP as model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import Preprocessing as beki_pre







#run
def fire(learning_rate, hide_num, nurans, epochs,bais,activation,Bounes):
    data = pd.read_csv("penguins.csv")
   # data = data[(data['species'] == class1) | (data['species'] == class2)]

    ##encoding and preprocessing
    labelEnconer = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    data.fillna('male', inplace=True)
    data['gender'] = labelEnconer.fit_transform(data['gender'])


    # normalize data
    scaler = MinMaxScaler()
    normalize_data = data.drop('species', axis=1, inplace=False)
    normalize = pd.DataFrame(scaler.fit_transform(normalize_data), columns=normalize_data.columns)
    data[normalize_data.columns] = normalize[normalize_data.columns].values


    ##class1
    C1 = data[(data['species'] == 'Adelie')]
    C1 = shuffle(C1)
    C1_y = C1['species']
    C1_x = C1.drop('species', axis=1, inplace=False)
    C1_x_train, C1_x_test, C1_y_train, C1_y_test = train_test_split(C1_x, C1_y, test_size=0.4, shuffle=True,
                                                                    random_state=7)

    ##class2
    C2 = data[data['species'] == 'Gentoo']
    C2 = shuffle(C2)
    C2_y = C2['species']
    C2_x = C2.drop('species', axis=1, inplace=False)
    C2_x_train, C2_x_test, C2_y_train, C2_y_test = train_test_split(C2_x, C2_y, test_size=0.4, shuffle=True,
                                                                    random_state=7)

    ##class3
    C3 = data[data['species'] == 'Chinstrap']
    C3 = shuffle(C3)
    C3_y = C3['species']

    C3_x = C3.drop('species', axis=1, inplace=False)
    C3_x_train, C3_x_test, C3_y_train, C3_y_test = train_test_split(C3_x, C3_y, test_size=0.4, shuffle=True,
                                                                    random_state=7)
    ## data
    x_train = pd.concat([C1_x_train, C2_x_train,C3_x_train])
    y_train = pd.concat([C1_y_train, C2_y_train,C3_y_train])
    x_test = pd.concat([C1_x_test, C2_x_test,C3_x_test])
    y_test = pd.concat([C1_y_test, C2_y_test,C3_y_test])

    ##data suffle

    data_train = x_train.assign(species=y_train)
    data_test= x_test.assign(species=y_test)

    data_train=shuffle(data_train)
    data_test=shuffle(data_test)

    ##data final to run

    y_train =data_train['species']
    y_test = data_test['species']
    x_test =data_test.drop('species', axis=1, inplace=False)
    x_train = data_train.drop('species', axis=1, inplace=False)

    #incoding
   # y_test = labelEnconer.fit_transform(y_test)
   # y_train= labelEnconer.fit_transform(y_train)
    y_train.replace(['Adelie','Gentoo','Chinstrap'],[0,1,2],inplace=True)
    y_test.replace(['Adelie', 'Gentoo', 'Chinstrap'], [0, 1, 2], inplace=True)

    y_test =y_test.to_numpy().reshape(len(y_test), 1)
    y_train = y_train.to_numpy().reshape(len(y_train), 1)


    y_test= onehot_encoder.fit_transform(y_test)
    y_train = onehot_encoder.fit_transform(y_train)
    mod = model.MLP(number_itration=epochs,number_of_neurons=nurans,number_hidden_layer=hide_num,lr=learning_rate,with_bais=bais,activation_function=activation)
    mod.fit(pd.DataFrame.to_numpy(x_train),y_train)
    prediction = mod.predict(pd.DataFrame.to_numpy(x_test))
    prediction_train = mod.predict(pd.DataFrame.to_numpy(x_train))
    print(f'Accuracy Train: {mod.accuracy(y_train,prediction_train)}%')

    def get_name(a):
        a = a.reshape(1,3)


        class1 = np.array([1,0,0])
        class2 = np.array([0,1,0])
        class3 = np.array([0, 0, 1])

        value = class1 == a

        if np.alltrue(value):
            return 'Adelie'

        value = class2 == a
        if np.alltrue(value):
            return 'Chinstrap'

        value = class3 == a
        if np.alltrue(value):
            return 'Gentoo'


    y_actual = np.apply_along_axis(get_name, 1, y_test)
    y_predication = np.apply_along_axis(get_name, 1, prediction)
    show_Con_Matrix(y_actual,y_predication)



    return [mod.accuracy(y_actual=y_test ,y_predict= prediction)]


def show_Con_Matrix(y_actual,y_predication):

        class1= 'Adelie'
        class2 = 'Gentoo'
        class3 = 'Chinstrap'
        conf_matrix = confusion_matrix(y_actual, y_predication)
        cm_df = pd.DataFrame(conf_matrix, index=[class1, class2, class3], columns=[class1, class2, class3])
        plt.figure(figsize=(7, 7))
        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()




#fire(.01,1,[4],200,True,1)
