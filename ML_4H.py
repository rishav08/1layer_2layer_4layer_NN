#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")

class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 4, h3 = 4, h4 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train, header = None, na_values = [' ?','?','? ',' ? '], skip_blank_lines = True)
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        
# #         Check corelation and remove attribute with high corelations
#         temp = pd.DataFrame(self.X)
# #         print(temp.head())
#         correlations = temp.corr(method='pearson')
# #         print(correlations)

# #         print(correlations)
# #         print()
#         ilist = []
#         jlist = []
#         for i in range(correlations.shape[0]):
#             for j in range(correlations.shape[1]):
#                 if correlations[i][j] >= 0.90 and i != j:
#                     if (i not in jlist and j not in ilist):
#                         ilist.append(i)
#                         jlist.append(j)
#                         # print(" i={}, j={}, corr= {}".format(i,j,correlations[i][j]))
#                         temp = temp.drop(j, axis=1)
#         ncols = len(temp.columns)
#         self.X = temp.iloc[:, 0:ncols].values.reshape(nrows, ncols)        
        
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=1)
        self.X = self.X_train
        self.y = self.y_train
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, h3)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, h3))
        self.w34 = 2 * np.random.random((h3, h4)) - 1
        self.X34 = np.zeros((len(self.X), h3))
        self.delta34 = np.zeros((h3, h4))        
        self.w45 = 2 * np.random.random((h4, output_layer_size)) - 1
        self.X45 = np.zeros((len(self.X), h4))
        self.delta45 = np.zeros((h4, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "relu":
            self.__relu(self, x)
        elif activation == "tanh":
            self.__tanh(self, x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "relu":
            self.__relu_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def __relu(self, x):
        zeros = np.zeros(x.shape)
        return np.maximum(zeros,x)
    
    def __relu_derivative(self, x):
        x[x > 0] = 1
        x[x < 0] = 0
        return x
        
    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def __tanh_derivative(self, x):
        return (1 - x * x)
    
    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
#         fill missing values with the most frequent
#         for i in range(X.shape[1]):
#             X[i] = X[i].fillna(X[i].value_counts().idxmax())

#         remove rows with missing value
        X = X.dropna(how = 'any')
    
        X = X.drop_duplicates()
        
#         Encode the attributes with object type columns
        le = preprocessing.LabelEncoder()
        for i in range(X.shape[1]):
            if X[i].dtype == 'object':
                X[i] = le.fit_transform(X[i])
                
#         Scale the attributes between 0 - 1
        scaler = preprocessing.MinMaxScaler()
        X = scaler.fit_transform(X)
        
#         Normalize the attributes
        X = preprocessing.normalize(X)

        X = pd.DataFrame(X)
        
        return X

    # Below is the training function

    def train(self, my_activation = "sigmoid", max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation=my_activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation=my_activation)
            update_layer4 = learning_rate * self.X45.T.dot(self.deltaOut)                        
            update_layer3 = learning_rate * self.X34.T.dot(self.delta45)
            update_layer2 = learning_rate * self.X23.T.dot(self.delta34)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w45 += update_layer4
            self.w34 += update_layer3
            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input
#             print(np.sum(error))

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)
        print(self.w34)
        print(self.w45)

    def forward_pass(self, activation):
        # pass our inputs through our neural network
        if activation == "sigmoid":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            self.X34 = self.__sigmoid(in3)
            in4 = np.dot(self.X34, self.w34)
            self.X45 = self.__sigmoid(in4)
            in5 = np.dot(self.X45, self.w45)
            out = self.__sigmoid(in5)
        elif activation == "relu":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            self.X34 = self.__relu(in3)
            in4 = np.dot(self.X34, self.w34)
            self.X45 = self.__relu(in4)
            in5 = np.dot(self.X45, self.w45)
            out = self.__relu(in5)
        elif activation == "tanh":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            self.X34 = self.__tanh(in3)
            in4 = np.dot(self.X34, self.w34)
            self.X45 = self.__tanh(in4)
            in5 = np.dot(self.X45, self.w45)
            out = self.__tanh(in5)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)        
        self.compute_hidden_layer4_delta(activation)     
        self.compute_hidden_layer3_delta(activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))

        self.deltaOut = delta_output

     # TODO: Implement other activation functions
    
    def compute_hidden_layer4_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer4 = (self.deltaOut.dot(self.w45.T)) * (self.__sigmoid_derivative(self.X45))
        elif activation == "relu":
            delta_hidden_layer4 = (self.deltaOut.dot(self.w45.T)) * (self.__relu_derivative(self.X45))
        elif activation == "tanh":
            delta_hidden_layer4 = (self.deltaOut.dot(self.w45.T)) * (self.__tanh_derivative(self.X45))

        self.delta45 = delta_hidden_layer4
        
    # TODO: Implement other activation functions

    def compute_hidden_layer3_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer3 = (self.delta45.dot(self.w34.T)) * (self.__sigmoid_derivative(self.X34))
        elif activation == "relu":
            delta_hidden_layer3 = (self.delta45.dot(self.w34.T)) * (self.__relu_derivative(self.X34))
        elif activation == "tanh":
            delta_hidden_layer3 = (self.delta45.dot(self.w34.T)) * (self.__tanh_derivative(self.X34))

        self.delta34 = delta_hidden_layer3
        
    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.delta34.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "relu":
            delta_hidden_layer2 = (self.delta34.dot(self.w23.T)) * (self.__relu_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.delta34.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions

    def compute_input_layer_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "relu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, my_activation = "sigmoid", header = True):
        self.X = self.X_test
        self.y = self.y_test
        out = self.forward_pass(activation = my_activation)        
        error = 0.5 * np.power((out - self.y), 2)        
        
        return np.sum(error) 


if __name__ == "__main__":
    
#     train = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#     actvtn = "sigmoid"    
#     no_of_iter = 10000
#     learn_rate = 0.09
    
    if(len(sys.argv) != 5):
        sys.exit("Please give the required amount of arguments - <Dataset path>, <Activation function like sigmoid, tanh or relu>, <No. of iterations>, <Learning Rate>")
    else:
        train = sys.argv[1]
        actvtn = sys.argv[2]
        no_of_iter = int(sys.argv[3])
        learn_rate = float(sys.argv[4])
        if(actvtn not in ["sigmoid","relu","tanh"]):
             sys.exit("Activation function should be either sigmoid or tanh or relu")

    neural_network = NeuralNet(train)
    neural_network.train(my_activation = actvtn, max_iterations = no_of_iter, learning_rate = learn_rate)    
    testError = neural_network.predict(my_activation = actvtn)
    print("Test Error is equal : " + str(testError))


