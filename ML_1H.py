import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")

class NeuralNet:
    def __init__(self, train, header = True, h1 = 4):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train, header = None, na_values = [' ?','?','? ',' ? '], skip_blank_lines = True)
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
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,  random_state=1)
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
        self.w12 = 2 * np.random.random((h1, output_layer_size)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
   

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "relu":
            self.__relu(self, x)
        elif activation == "tanh":
            self.__tanh(self, x)

 

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
    


    def preprocess(self, X):
# #         fill missing values with the most frequent
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
            update_layer1 = learning_rate * self.X12.T.dot(self.deltaOut)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w12 += update_layer1
            self.w01 += update_input
#             print(np.sum(error))/
        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)

    def forward_pass(self, activation):
        # pass our inputs through our neural network
        if activation == "sigmoid":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            out = self.__sigmoid(in2)
        elif activation == "relu":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            out = self.__relu(in2)
        elif activation == "tanh":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            out = self.__tanh(in2)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer1_delta(activation)


    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))

        self.deltaOut = delta_output



    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.deltaOut.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "relu":
            delta_hidden_layer1 = (self.deltaOut.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.deltaOut.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    def compute_input_layer_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "relu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    # Assumed that the test dataset has the same format as the training dataset

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
#     learn_rate = 0.0009
    
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
    

