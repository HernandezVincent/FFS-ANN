import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import linear_model
from sklearn.metrics import r2_score

def get_data(test):

    directory = "Data\\" + test + "\\"

    name = "data_X.npy"
    data_X = np.load(directory + name)
    name = "Mean_X.npy"
    mean_X = np.load(directory + name)
    name = "Std_X.npy"
    std_X = np.load(directory + name)
    name = "name_X.npy"
    name_X = np.load(directory + name)

    # Standardize input
    for i in range(data_X.shape[1]):
        data_X[:, i] = (data_X[:, i] - mean_X[i]) / std_X[i]

    name = "data_Y.npy"
    data_Y = np.load(directory + name)
    name = "Mean_Y.npy"
    mean_Y = np.load(directory + name)
    name = "Std_Y.npy"
    std_Y = np.load(directory + name)
    name = "name_Y.npy"
    name_Y = np.load(directory + name)

    return data_X, mean_X, std_X, name_X, data_Y, mean_Y, std_Y, name_Y

def scatter_regresion_Plot(X, Y, test):

    plt.scatter(X, Y, c = 'b', label = '_nolegend_', s = 1)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    R2 = r2_score(X, Y)

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    plt.plot(X, regr.predict(X), "--", label = 'Regression', color = 'r')
    plt.title(test + ' ($R^2$: ' + "{0:.3f}".format(R2) + ")", fontsize = 14)
    plt.xlabel('True Values', fontsize = 12, weight = 'bold')
    plt.ylabel('Predicted Values', fontsize = 12, weight = 'bold')
    plt.legend(loc = 'upper left', bbox_to_anchor = (0, 1.0), fancybox = True, shadow = True, fontsize = 10)
    plt.subplots_adjust(left = 0.2, right = 0.9, bottom = 0.05, top = 0.97, wspace = 0.15, hspace = 0.3)

    return R2

class NN_Class:

    def __init__(NN, test):

        NN.testName = test

    def set_data(NN, data_X, mean_X, std_X, name_X, data_Y, mean_Y, std_Y, name_Y):

        NN.data_X = data_X
        NN.mean_X = mean_X
        NN.std_X = std_X
        NN.name_X = name_X
        NN.data_Y = data_Y
        NN.mean_Y = mean_Y
        NN.std_Y = std_Y
        NN.name_Y = name_Y

        NN.InputSize = NN.data_X.shape[1]
        NN.OutputSize = NN.data_Y.shape[1]

    def build_network(NN):

        # Create graph and session
        tf.reset_default_graph()

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        NN.sess = tf.Session(config=config)
        NN.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope(tf.get_variable_scope()):

            NN.x = tf.placeholder('float32', [None, NN.InputSize])  # Input Tensor
            NN.y = tf.placeholder('float32', [None, NN.OutputSize])  # Output Tensor
            print(NN.x)

            with tf.name_scope('Dense'):
                with tf.variable_scope('dense_layer_0', reuse=False):
                    w = tf.get_variable("w", shape=[NN.InputSize, 512], initializer=tf.random_normal_initializer(mean=0., stddev=0.1))
                    b = tf.get_variable("b", shape=[512], initializer=tf.constant_initializer(0.0))
                    l = tf.add(tf.matmul(NN.x, w), b)
                    l = tf.nn.sigmoid(l)
                    print(l)
                    l = tf.nn.dropout(l, NN.keep_prob)
                    print(l)

                with tf.variable_scope('dense_layer_1', reuse=False):
                    w = tf.get_variable("w", shape=[512, 1024], initializer=tf.random_normal_initializer(mean=0., stddev=0.1))
                    b = tf.get_variable("b", shape=[1024], initializer=tf.constant_initializer(0.0))
                    l = tf.add(tf.matmul(l, w), b)
                    l = tf.nn.sigmoid(l)
                    print(l)
                    l = tf.nn.dropout(l, NN.keep_prob)
                    print(l)

                with tf.variable_scope('dense_layer_2', reuse=False):
                    w = tf.get_variable("w", shape=[1024, 512], initializer=tf.random_normal_initializer(mean=0., stddev=0.1))
                    b = tf.get_variable("b", shape=[512], initializer=tf.constant_initializer(0.0))
                    l = tf.add(tf.matmul(l, w), b)
                    l = tf.nn.sigmoid(l)
                    print(l)
                    l = tf.nn.dropout(l, NN.keep_prob)
                    print(l)

                with tf.variable_scope('dense_layer_3', reuse=False):
                    w = tf.get_variable("w", shape=[512, NN.OutputSize], initializer=tf.random_normal_initializer(mean=0., stddev=0.1))
                    b = tf.get_variable("b", shape=[NN.OutputSize], initializer=tf.constant_initializer(0.0))
                    NN.y_ = tf.add(tf.matmul(l, w), b)
                    print(NN.y_)

        NN.saver = tf.train.Saver(save_relative_paths=True)

    def test(NN):

        NN.build_network()

        save_folder = ".\\" + "ANN\\" + NN.testName + "\\"
        NN.sess.run(tf.global_variables_initializer())
        NN.saver.restore(NN.sess, save_folder)

        test_data = {NN.x: NN.data_X, NN.keep_prob: 1}
        predictedValues_N = NN.y_.eval(feed_dict=test_data, session=NN.sess)

        pred = predictedValues_N.copy()
        for i in range(predictedValues_N.shape[1]):
            pred[:, i] = (predictedValues_N[:, i] * NN.std_Y[i]) + NN.mean_Y[i]

        true = NN.data_Y

        return pred, true

tests = ['Index', 'Vertex45', 'Vertex35', 'Vertex25', 'Vertex15']

for test in tests:
    if test == 'Index':
        fig = plt.figure(figsize=(14, 10))
        IndexName = ['Maximal force (N)', 'Volume $N^3$', 'Isotropy', 'Main Axis X', 'Main Axis Y', 'Main Axis Z']
    elif test == 'Vertex45':
        fig = plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1)
    elif test == 'Vertex35':
        plt.subplot(2, 2, 2)
    elif test == 'Vertex25':
        plt.subplot(2, 2, 3)
    elif test == 'Vertex15':
        plt.subplot(2, 2, 4)

    ########################### Load NN ############################
    data_X, mean_X, std_X, name_X, data_Y, mean_Y, std_Y, name_Y = get_data(test=test)
    NN = NN_Class(test=test)
    NN.set_data(data_X, mean_X, std_X, name_X, data_Y, mean_Y, std_Y, name_Y)
    pred, true = NN.test()

    ########################### Regression and plot ###################################
    if test == 'Index':
        for i in range(len(IndexName)):
            plt.subplot(3, 2, i + 1)
            scatter_regresion_Plot(true[:, i], pred[:, i], IndexName[i])
            axes = plt.gca()
    else:
        scatter_regresion_Plot(true[:, :], pred[:, :], test)
        axes = plt.gca()
        axes.set_xlim([-10, 1000])
        axes.set_ylim([-10, 1000])
