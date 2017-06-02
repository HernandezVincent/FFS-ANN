import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import tensorflow as tf
from itertools import chain
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score
import sys, time
from mpl_toolkits.mplot3d import Axes3D
import os, math
from tabulate import tabulate
import scipy
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import random
import numpy_indexed as npi
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt

def get_data(TestName):

    df = pd.read_csv('Data/TestData.csv', sep = ';')
    df.dropna(how = 'any')
    InPutName = ['shoulder0_X', 'shoulder0_Y', 'shoulder0_Z', 'elbow_X', 'elbow_Y', 'elbow_Z',
                 'wrist_hand_X', 'wrist_hand_Y', 'wrist_hand_Z', 'EF_X', 'EF_Y', 'EF_Z', 'SubjectMass']
    dataInput = df[InPutName].as_matrix()

    if TestName == 'Index':
        OutPutName = ['Max', 'Volume', 'Isotropy', 'MainAxis_X', 'MainAxis_Y', 'MainAxis_Z']
    else:
        if TestName == 'Vertex45':
            angle = 45
            nVertex = 26
        elif TestName == 'Vertex35':
            angle = 35
            nVertex = 56
        elif TestName == 'Vertex25':
            angle = 25
            nVertex = 106
        elif TestName == 'Vertex15':
            angle = 15
            nVertex = 266

        OutPutName = list(chain.from_iterable([['Theta_' + str(angle) + '_V' + str(s + 1) + '_N'] for s in range(nVertex)]))

    dataOutput = df[OutPutName].as_matrix()

    return dataInput, dataOutput, OutPutName, InPutName

def normalize_data(data):
    std = []
    mean = []
    dataOut = data.copy()

    for i in range(0, len(data[0])):
        std.append(np.std(data[:, i], ddof = 1))
        mean.append(np.mean(data[:, i]))
        dataOut[:, i] = (data[:, i] - mean[i]) / std[i]

    dataOut[np.isnan(dataOut)] = 0

    return dataOut, mean, std

def denormalize_data(data, mean, std):
    dataOut = data.copy()

    for i in range(0, len(data[0])):
        dataOut[:, i] = (data[:, i] * std[i]) + mean[i]

    return dataOut

def normalize_data2(data, mean, std):

    dataOut = data.copy()

    for i in range(0, len(data[0])):
        dataOut[:, i] = (data[:, i] - mean[i]) / std[i]

    return dataOut

def scatter_Regresion_Plot(X, Y, TestName):

    plt.scatter(X, Y, c = 'b', label = '_nolegend_', s = 1)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    R2 = r2_score(X, Y)

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    plt.plot(X, regr.predict(X), "--", label = 'Regression', color = 'r')
    plt.title(TestName + ' ($R^2$: ' + "{0:.3f}".format(R2) + ")", fontsize = 14)
    plt.xlabel('True Values', fontsize = 12, weight = 'bold')
    plt.ylabel('Predicted Values', fontsize = 12, weight = 'bold')
    plt.legend(loc = 'upper left', bbox_to_anchor = (0, 1.0), fancybox = True, shadow = True, fontsize = 10)
    plt.subplots_adjust(left = 0.2, right = 0.9, bottom = 0.05, top = 0.97, wspace = 0.15, hspace = 0.3)

class NN_Class:

    def __init__(self, SaveFolder = 'ANN/', TestName = 'Vertex15'):

        self.BaseSaveFolder = SaveFolder
        self.TestName = TestName

    def set_NN_properties(self, InputSize, LayersSize, OutputSize, ActivationFunction = 'sigmoid'):

        self.InputSize = InputSize
        self.OutputSize = OutputSize
        self.LayersSize = LayersSize
        self.N_Layers = len(LayersSize)
        self.ActivationFunction = ActivationFunction
        self.Topology = [[InputSize, LayersSize, OutputSize], [ActivationFunction]]

        self.SaveFolder = self.BaseSaveFolder + "NN_" + '_'.join(
            str(LayersSize[i]) for i in range(self.N_Layers)) + "_" + '_'.join(
            ActivationFunction) + '_' + self.TestName + '/'
        self.ANNPath = self.SaveFolder + 'ANN'
        self.MeanStdInputPath = self.SaveFolder + 'MeanStdInput' + self.TestName + '.csv'
        self.MeanStdOutputPath = self.SaveFolder + 'MeanStdOutput' + self.TestName + '.csv'

        print('Topology: ' + str(self.Topology) + ' - Saved in / loaded from: ' + self.SaveFolder)

    def load_neural_network(self):

        MeanStdInput = pd.read_csv(self.MeanStdInputPath, sep = ',').set_index('Unnamed: 0').as_matrix()
        self.meanInput = np.array(MeanStdInput[0])
        self.stdInput = np.array(MeanStdInput[1])
        MeanStdOutput = pd.read_csv(self.MeanStdOutputPath, sep = ',').set_index('Unnamed: 0').as_matrix()
        self.meanOutput = np.array(MeanStdOutput[0])
        self.stdOutput = np.array(MeanStdOutput[1])

        tf.reset_default_graph()

        with tf.Graph().as_default(), tf.Session() as self.sess:

            self.x = tf.placeholder('float32', [None, self.InputSize])  # Input Tensor
            self.y_ = tf.placeholder('float32', [None, self.OutputSize])  # Output Tensor for the known values
            self.create_NN()
            self.sess.run(tf.global_variables_initializer())
            self.sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
            saver = tf.train.Saver()
            saver = tf.train.import_meta_graph(self.ANNPath + '.meta')
            saver.restore(self.sess, self.ANNPath)
            print('Artificial Neural Network from: ' + self.SaveFolder + ' loaded !')

    def create_NN(self):

        l = self.create_layers(self.x, self.InputSize, self.LayersSize[0], activation = self.ActivationFunction[0])
        if self.N_Layers > 1:
            for i in range(self.N_Layers - 1):
                l = self.create_layers(l, self.LayersSize[i - 1], self.LayersSize[i],
                                       activation = self.ActivationFunction[i])

        self.y = self.create_layers(l, self.LayersSize[-1], self.OutputSize, activation = 'None')

    def create_layers(self, inputs, in_size, out_size, activation = 'sigmoid'):

        w = tf.Variable(tf.random_normal([in_size, out_size]))
        b = tf.Variable(tf.random_normal([out_size]))
        l = tf.add(tf.matmul(inputs, w), b)

        if activation == 'sigmoid':
            l = tf.nn.sigmoid(l)
        else:
            l = l
        return l

    def use_neural_network(self, use_x):
        Input_data = {self.x: use_x}
        predictions = self.sess.run(self.y, feed_dict = Input_data)
        use_y = np.array(predictions).astype('float32')

        use_y = denormalize_data(use_y, self.meanOutput, self.stdOutput)

        return use_y

Tests = ['Index', 'Vertex45', 'Vertex35', 'Vertex25', 'Vertex15']
for test in Tests:
    InputSize = 13
    if test == 'Index':
        fig = plt.figure(figsize = (14, 10))
        OutputSize = 6
        LayersSize = [50, 50, 50]
        ActivationFunction = ['sigmoid', 'sigmoid', 'sigmoid']
        IndexName = ['Maximal force (N)', 'Volume $N^3$', 'Isotropy', 'Main Axis X', 'Main Axis Y', 'Main Axis Z']
    elif test == 'Vertex45':
        fig = plt.figure(figsize = (14, 10))
        plt.subplot(2, 2, 1)
        OutputSize = 26
        LayersSize = [100, 100, 100]
        ActivationFunction = ['sigmoid', 'sigmoid', 'sigmoid']
    elif test == 'Vertex35':
        plt.subplot(2, 2, 2)
        OutputSize = 56
        LayersSize = [200, 200, 200]
        ActivationFunction = ['sigmoid', 'sigmoid', 'sigmoid']
    elif test == 'Vertex25':
        plt.subplot(2, 2, 3)
        OutputSize = 106
        LayersSize = [100, 100, 100]
        ActivationFunction = ['sigmoid', 'sigmoid', 'sigmoid']
    elif test == 'Vertex15':
        plt.subplot(2, 2, 4)
        OutputSize = 266
        LayersSize = [200, 200, 200]
        ActivationFunction = ['sigmoid', 'sigmoid', 'sigmoid']

    ########################### Load NN ############################

    dataInput, dataOutput, OutPutName, InPutName = get_data(TestName = test)
    InputSize, n_samples, OutputSize = len(dataInput[0]), len(dataInput), len(dataOutput[0])
    NN = NN_Class(SaveFolder = 'ANN/', TestName = test)
    NN.set_NN_properties(InputSize, LayersSize, OutputSize, ActivationFunction)
    NN.load_neural_network()
    X = normalize_data2(dataInput, NN.meanInput, NN.stdInput)
    Y_ = dataOutput
    Y = NN.use_neural_network(X)

    ########################### Regression and plot ###################################

    if test == 'Index':
        for i in range(len(IndexName)):
            plt.subplot(3, 2, i + 1)
            scatter_Regresion_Plot(Y_[:, i], Y[:, i], IndexName[i])
            axes = plt.gca()

    else:
        scatter_Regresion_Plot(Y[:, :], Y_[:, :], test)
        axes = plt.gca()
        axes.set_xlim([-10, 1000])
        axes.set_ylim([-10, 1000])
