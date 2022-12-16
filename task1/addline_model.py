import numpy as np
class Model:

    def __init__(self, itration=100,learning_rate=0.01, withBaise = True):
        self.learning_rate = learning_rate
        self.itration = itration
        self.withBaise = withBaise
        self.w = None
        self.b = None


    def fit(self, X, Y):
        number_feature = X.shape[1]

        self.w = np.random.rand(number_feature)
        self.b = 0
        for index in range(self.itration):
            for i, x in enumerate(X):

                if not self.withBaise:
                    sumtion = np.dot(self.w.T, x)
                else:
                    sumtion = np.dot(self.w.T, x) + self.b

                predict_value = self.activation_function(sumtion)

                if Y[i] != predict_value:
                    diffrence = Y[i] - predict_value
                    update_w = self.learning_rate * diffrence
                    self.w = self.w + update_w * x

                    if self.withBaise:
                        self.b = self.b + update_w

    def predict(self, X):
        if not self.withBaise:
            sumtion = np.dot(X, self.w)
        else:
            sumtion = np.dot(X, self.w) + self.b

        Y_prediction = self.activation_function(sumtion)
        return Y_prediction

    def activation_function(self, X):

        o = np.where(X > 0, 1, -1)
        '''
        for i in range(len(X.shape[0])):
            if X[i] > 0:
                o[i] = 1
            else:
                o[i] = -1
        '''
        return o


    def matrix_con(self, y_prediction, y_actual):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(y_prediction)):
            if y_prediction[i] != y_actual[i]:
                if y_actual[i] > 0:
                    FP += 1
                elif y_actual[i] < 0:
                    FN += 1
            else:
                if y_actual[i] > 0:
                    TP += 1
                elif y_actual[i] < 0:
                    TN += 1

        return TP, TN, FP, FN

    def get_accuracy(self, Y_actual, Y_prediction):
        sum = 0
        for i in range(len(Y_actual)):
            if(Y_actual[i] == Y_prediction[i]):
                sum+=1
        sum /= float(len(Y_actual))

        return sum*100


    def weight(self):
        if(self.withBaise):
            return [self.w[0],self.w[1],self.b]
        else:
         return self.w
