import  numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

class MLP:
    def __init__(self, number_itration=1, number_hidden_layer=3, number_of_neurons=[], lr=0.01, with_bais=True,
                 activation_function=1):
        self.number_itration = number_itration
        self.number_hidden_layer = number_hidden_layer
        self.number_of_neurons = number_of_neurons
        self.withBaise = with_bais
        self.lr = lr
        self.weights = []
        self.baises = []
        self.activation_function = activation_function

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def Hyperbolic_tangent(self,x):
        return np.tanh(x)



    def initlization(self, no_feature, no_calsses):
        inpute_layer = no_feature
        output_layer = no_calsses
        first_hidden_layer = self.number_of_neurons[0]
        last_hidden_layer = self.number_of_neurons[self.number_hidden_layer - 1]

        self.weights.append(np.random.rand(first_hidden_layer, inpute_layer))
        self.baises.append(np.ones(shape=(first_hidden_layer, 1)))

        for i in range(1, self.number_hidden_layer):
            current_layer = self.number_of_neurons[i]
            last_layer = self.number_of_neurons[i - 1]
            self.weights.append(np.random.rand(current_layer, last_layer))
            self.baises.append(np.ones(shape=(current_layer, 1)))

        self.weights.append(np.random.rand(output_layer, last_hidden_layer))
        self.baises.append(np.ones(shape=(output_layer, 1)))

    def feedForward(self, x, y):
        all_output_from_neurons = []
        inpute = x.T
        all_output_from_neurons.append(inpute)

        for i in range(self.number_hidden_layer + 1):

            if self.withBaise:
                Z = np.dot(self.weights[i], inpute) + self.baises[i]

            else:
                Z = np.dot(self.weights[i], inpute)

            if self.activation_function == 1:
                A = self.sigmoid(Z)
            else:
                A = self.Hyperbolic_tangent(Z)


            all_output_from_neurons.append(A)

            inpute = A

        error = y - A
        if self.activation_function == 1:
            local_gradient = error * (A) * (1 - A)
        else:
            local_gradient = error * ( 1 - (A*A) )

        return local_gradient, all_output_from_neurons

    def backWord(self, local_gradient, all_outputs):

        list_local_gradinet = []
        list_local_gradinet.append(local_gradient)
        for i in reversed(range(1, len(self.weights))):
            if self.activation_function == 1:
                da = (all_outputs[i]) * (1 - all_outputs[i])
            else:
                da = 1 - (all_outputs[i] * all_outputs[i])


            local_gradient_neuron = np.dot(self.weights[i].T, local_gradient) * da
            local_gradient = local_gradient_neuron
            list_local_gradinet.append(local_gradient_neuron)

        list_local_gradinet.reverse()

        return list_local_gradinet

    def feedForwardUpdate(self, list_local_gradinet, outputs_neurons):

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.lr * (np.dot(list_local_gradinet[i], outputs_neurons[i].T))
            if self.withBaise:
                self.baises[i] = self.baises[i] + list_local_gradinet[i]

    def fit(self, X, Y):
        m = X.shape[0]
        no_feature = X.shape[1]
        no_classes = 3

        self.initlization(no_feature=X.shape[1], no_calsses=no_classes)

        for _ in range(self.number_itration):
            for i, x_i in enumerate(X):
                local_gradient, all_outputs = self.feedForward(x_i.reshape(1, no_feature), Y[i].reshape(no_classes, 1))
                list_local_gradient = self.backWord(local_gradient, all_outputs)
                self.feedForwardUpdate(list_local_gradinet=list_local_gradient, outputs_neurons=all_outputs)

    def predict(self, X):
        m = X.shape[0]
        no_feature = X.shape[1]
        output = np.zeros(shape=(m, 3))

        for j, x_i in enumerate(X):
            x = x_i.reshape(no_feature, 1)

            for i in range(self.number_hidden_layer + 1):
                if self.withBaise:

                    Z = np.dot(self.weights[i], x) + self.baises[i]
                else:
                    Z = np.dot(self.weights[i], x)

                if self.activation_function == 1:
                    A = self.sigmoid(Z)
                else:
                    A = self.Hyperbolic_tangent(Z)
                x = A
            output[j] = A.reshape(3,)

        for i in range(len(output)):
            output[i] = np.where(output[i] >= output[i].max(), 1, 0)

        return output

    def accuracy(self,y_actual , y_predict):
        m = y_actual.shape[0]
        output = np.random.rand(m, 1)
        value = (y_actual == y_predict)
        # print(value)
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
        for i in range(m):
            output[i] = np.where(np.alltrue(value[i]), 1, 0)

        # print(output)
        return ((sum(output)[0] / m) * 100)





    '''def ConfusionMatrix(self,y_test, predictions, class_1, class_2, class_3):
        cm = confusion_matrix(y_test, predictions)
        cm_df = pd.DataFrame(cm,index = [class_1,class_2,class_3],
                                columns = [class_1,class_2,class_3])
        plt.figure(figsize=(6,6))
        sns.heatmap(cm_df, annot=True)

        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()'''
