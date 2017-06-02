import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import chain
from sklearn import linear_model
from sklearn.metrics import r2_score

def get_data(testName):

    df = pd.read_csv('Data/TestData.csv', sep = ';')
    df.dropna(how = 'any')
    inPutName = ['shoulder0_X', 'shoulder0_Y', 'shoulder0_Z', 'elbow_X', 'elbow_Y', 'elbow_Z',
                 'wrist_hand_X', 'wrist_hand_Y', 'wrist_hand_Z', 'EF_X', 'EF_Y', 'EF_Z', 'SubjectMass']
    dataInput = df[inPutName].as_matrix()

    if testName == 'Index':
        outPutName = ['Max', 'Volume', 'Isotropy', 'MainAxis_X', 'MainAxis_Y', 'MainAxis_Z']
    else:
        if testName == 'Vertex45':
            angle = 45
            nVertex = 26
        elif testName == 'Vertex35':
            angle = 35
            nVertex = 56
        elif testName == 'Vertex25':
            angle = 25
            nVertex = 106
        elif testName == 'Vertex15':
            angle = 15
            nVertex = 266

        outPutName = list(chain.from_iterable([['Theta_' + str(angle) + '_V' + str(s + 1) + '_N'] for s in range(nVertex)]))

    dataOutput = df[outPutName].as_matrix()

    return dataInput, dataOutput, outPutName, inPutName

def standardize_data(data):
    std = []
    mean = []
    dataOut = data.copy()

    for i in range(0, len(data[0])):
        std.append(np.std(data[:, i], ddof = 1))
        mean.append(np.mean(data[:, i]))
        dataOut[:, i] = (data[:, i] - mean[i]) / std[i]

    dataOut[np.isnan(dataOut)] = 0

    return dataOut, mean, std

def reverse_standardize_data(data, mean, std):
    dataOut = data.copy()

    for i in range(0, len(data[0])):
        dataOut[:, i] = (data[:, i] * std[i]) + mean[i]

    return dataOut

def standardize_data_2(data, mean, std):

    dataOut = data.copy()

    for i in range(0, len(data[0])):
        dataOut[:, i] = (data[:, i] - mean[i]) / std[i]

    return dataOut

def scatter_regresion_Plot(X, Y, testName):

    plt.scatter(X, Y, c = 'b', label = '_nolegend_', s = 1)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    R2 = r2_score(X, Y)

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    plt.plot(X, regr.predict(X), "--", label = 'Regression', color = 'r')
    plt.title(testName + ' ($R^2$: ' + "{0:.3f}".format(R2) + ")", fontsize = 14)
    plt.xlabel('True Values', fontsize = 12, weight = 'bold')
    plt.ylabel('Predicted Values', fontsize = 12, weight = 'bold')
    plt.legend(loc = 'upper left', bbox_to_anchor = (0, 1.0), fancybox = True, shadow = True, fontsize = 10)
    plt.subplots_adjust(left = 0.2, right = 0.9, bottom = 0.05, top = 0.97, wspace = 0.15, hspace = 0.3)

class NN_Class:

    def __init__(self, rootFolder = 'ANN/', testName = 'Vertex15'):

        self.rootFolder = rootFolder
        self.testName = testName

    def set_NN_properties(self, inputSize, layersSize, outputSize, activationFunction = 'sigmoid'):

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.layersSize = layersSize
        self.N_Layers = len(layersSize)
        self.activationFunction = activationFunction
        self.topology = [[inputSize, layersSize, outputSize], [activationFunction]]

        self.saveFolder = self.rootFolder + "NN_" + '_'.join(
            str(layersSize[i]) for i in range(self.N_Layers)) + "_" + '_'.join(
            activationFunction) + '_' + self.testName + '/'
        self.ANNPath = self.saveFolder + 'ANN'
        self.meanStdInputPath = self.saveFolder + 'meanStdInput' + self.testName + '.csv'
        self.meanStdOutputPath = self.saveFolder + 'meanStdOutput' + self.testName + '.csv'

        print('topology: ' + str(self.topology) + ' - Saved in / loaded from: ' + self.saveFolder)

    def load_neural_network(self):

        meanStdInput = pd.read_csv(self.meanStdInputPath, sep = ',').set_index('Unnamed: 0').as_matrix()
        self.meanInput = np.array(meanStdInput[0])
        self.stdInput = np.array(meanStdInput[1])
        meanStdOutput = pd.read_csv(self.meanStdOutputPath, sep = ',').set_index('Unnamed: 0').as_matrix()
        self.meanOutput = np.array(meanStdOutput[0])
        self.stdOutput = np.array(meanStdOutput[1])

        tf.reset_default_graph()

        with tf.Graph().as_default(), tf.Session() as self.sess:

            self.x = tf.placeholder('float32', [None, self.inputSize])  # Input Tensor
            self.y_ = tf.placeholder('float32', [None, self.outputSize])  # Output Tensor
            self.create_NN()
            self.sess.run(tf.global_variables_initializer())
            self.sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
            saver = tf.train.Saver()
            saver = tf.train.import_meta_graph(self.ANNPath + '.meta')
            saver.restore(self.sess, self.ANNPath)
            print('Artificial Neural Network from: ' + self.saveFolder + ' loaded !')

    def create_NN(self):

        l = self.create_layers(self.x, self.inputSize, self.layersSize[0], activation = self.activationFunction[0])
        if self.N_Layers > 1:
            for i in range(self.N_Layers - 1):
                l = self.create_layers(l, self.layersSize[i - 1], self.layersSize[i],
                                       activation = self.activationFunction[i])

        self.y = self.create_layers(l, self.layersSize[-1], self.outputSize, activation = 'None')

    def create_layers(self, inputs, in_size, out_size, activation = 'sigmoid'):

        w = tf.Variable(tf.random_normal([in_size, out_size]))
        b = tf.Variable(tf.random_normal([out_size]))
        l = tf.add(tf.matmul(inputs, w), b)

        if activation == 'sigmoid':
            l = tf.nn.sigmoid(l)
        else:
            l = l
        return l

    def use_neural_network(self, X):

        Input_data = {self.x: X}
        predictions = self.sess.run(self.y, feed_dict = Input_data)
        predicted_Y = np.array(predictions).astype('float32')
        predicted_Y = reverse_standardize_data(predicted_Y, self.meanOutput, self.stdOutput)

        return predicted_Y

tests = ['Index', 'Vertex45', 'Vertex35', 'Vertex25', 'Vertex15']
for test in tests:
    inputSize = 13
    if test == 'Index':
        fig = plt.figure(figsize = (14, 10))
        outputSize = 6
        layersSize = [50, 50, 50]
        activationFunction = ['sigmoid', 'sigmoid', 'sigmoid']
        IndexName = ['Maximal force (N)', 'Volume $N^3$', 'Isotropy', 'Main Axis X', 'Main Axis Y', 'Main Axis Z']
    elif test == 'Vertex45':
        fig = plt.figure(figsize = (14, 10))
        plt.subplot(2, 2, 1)
        outputSize = 26
        layersSize = [100, 100, 100]
        activationFunction = ['sigmoid', 'sigmoid', 'sigmoid']
    elif test == 'Vertex35':
        plt.subplot(2, 2, 2)
        outputSize = 56
        layersSize = [200, 200, 200]
        activationFunction = ['sigmoid', 'sigmoid', 'sigmoid']
    elif test == 'Vertex25':
        plt.subplot(2, 2, 3)
        outputSize = 106
        layersSize = [100, 100, 100]
        activationFunction = ['sigmoid', 'sigmoid', 'sigmoid']
    elif test == 'Vertex15':
        plt.subplot(2, 2, 4)
        outputSize = 266
        layersSize = [200, 200, 200]
        activationFunction = ['sigmoid', 'sigmoid', 'sigmoid']

    ########################### Load NN ############################

    dataInput, dataOutput, outPutName, inPutName = get_data(testName = test)
    inputSize, n_samples, outputSize = len(dataInput[0]), len(dataInput), len(dataOutput[0])
    NN = NN_Class(rootFolder = 'ANN/', testName = test)
    NN.set_NN_properties(inputSize, layersSize, outputSize, activationFunction)
    NN.load_neural_network()

    X = standardize_data_2(dataInput, NN.meanInput, NN.stdInput)
    Y_true = dataOutput
    Y_predicted = NN.use_neural_network(X)

    ########################### Regression and plot ###################################

    if test == 'Index':
        for i in range(len(IndexName)):
            plt.subplot(3, 2, i + 1)
            scatter_regresion_Plot(Y_true[:, i], Y_predicted[:, i], IndexName[i])
            axes = plt.gca()
    else:
        scatter_regresion_Plot(Y_true[:, :], Y_predicted[:, :], test)
        axes = plt.gca()
        axes.set_xlim([-10, 1000])
        axes.set_ylim([-10, 1000])