"""
  Author    Byunghyun Ban
  Email     halfbottle@sangsang.farm
  CTO of Imagination Garden Inc.
  http://sangsang.farm
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time


def csv_into_array(filename):
    file = open(filename)
    header = file.readline().strip().split(", ")
    file.close()
    data = np.genfromtxt(filename, delimiter=", ", skip_header=1)
    if len(header) == 1:
        data = data.reshape((len(data), 1))
    return header, data


def make_equation(function, slopes, intercepts):
    equations = []
    for i in range(len(slopes)):
        equations.append(Equation(function, slopes[i], intercepts[i]))
    return equations


class Equation:
    def __init__(self, function, slopes, intercept):
        self.function = function
        self.slopes = slopes
        self.intercept = intercept

    def __str__(self):
        if self.function == "quadratic":
            return "Y = " + str(self.slopes[0]) + " X^2 + " + str(self.slopes[1]) + " X " + str(self.intercept)
        elif self.function == "linear":
            return "Y = " + str(self.slopes[0]) + " X + " + str(self.intercept)

    def calculate(self, x):
        if self.function == "quadratic":
            return self.slopes[0] * x**2 + self.slopes[1] * x + self.intercept
        elif self.function == "linear":
            return self.slopes[0] * x + self.intercept


class Linear:
    def __init__(self, data_filename, label_filename):
        self.data_header, self.data = csv_into_array(data_filename)
        self.label_header, self.label = csv_into_array(label_filename)

        if self.data.shape != self.label.shape:
            print("The data and label has different shape.")
            exit(1)

        self.data_size, self.data_category = self.data.shape

        self.slopes = []
        self.intercepts = []

        self.regression(self.data, self.label) # x y 모두 그대로 선형회귀
        self.equation = make_equation("linear", self.slopes, self.intercepts)

    def regression(self, data, label):
        for i in range(len(self.data_category)):
            regression_result = np.polyfit(data[:, i], label[:, i], 1)
            self.slopes.append(regression_result[0])
            self.intercepts.append(regression_result[1])

        report = "Regression Result\n"
        for i in range(len(self.slopes)):
            report += "Regression number " + str(i + 1) + ".\n"
            report += "Slope : " + str(self.slopes[i]) + "\n"
            report += "Intercept : " + str(self.intercepts[i]) + "\n\n\n"

        report_file = open("Report_" + str(time.time()) + ".txt", "w")
        report_file.write(report)
        report_file.close()
        print(report)

    def refine_concentration(self, filename):
        header, data = csv_into_array(filename)
        for i in range(len(self.equation)):
            data[:, i] = self.equation[i].calculate(data[:, i])
        np.savetxt("converted_" + filename, data, delimiter=", ", header=str(header))


class Quadratic:
    def __init__(self, data_filename, label_filename):
        self.data_header, self.data = csv_into_array(data_filename)
        self.label_header, self.label = csv_into_array(label_filename)

        if self.data.shape != self.label.shape:
            print("The data and label has different shape.")
            exit(1)

        self.data_size, self.data_category = self.data.shape

        self.slopes = []
        self.intercepts = []

        self.quadratic_regression(self.data, self.label) # x y 둘다 그대로 quadratic regression
        self.equation = make_equation("quadratic", self.slopes, self.intercepts)

    def quadratic_regression(self, data, label):
        for i in range(len(self.data_category)):
            regression_result = np.polyfit(data[:, i], label[:, i], 2)
            self.slopes.append(regression_result[0:2])
            self.intercepts.append(regression_result[2])

        report = "Regression Result\n"
        for i in range(len(self.slopes)):
            report += "Regression number " + str(i + 1) + ".\n"
            report += "Slope : " + str(self.slopes[i]) + "\n"
            report += "Intercept : " + str(self.intercepts[i]) + "\n\n\n"

        report_file = open("Report_" + str(time.time()) + ".txt", "w")
        report_file.write(report)
        report_file.close()
        print(report)

    def volt_to_concentration(self, filename):
        header, data = csv_into_array(filename)
        for i in range(len(self.equation)):
            data[:, i] = self.equation[i].calculate(data[:, i])
        np.savetxt("converted_" + filename, data, delimiter=", ", header=str(header))


class DeepLearning:
    def __init__(self, data_filename, label_filename):
        self.batch_size = 128
        self.epoch = 10
        self.device = "/gpu:0"
        self.optimizer = "adam"
        self.loss = "mean_squared_error"
        # self.loss = "mean_absolute_error"
        # self.loss = "mean_absolute_percentage_error"
        # self.loss = "mean_squared_logarithmic_error"

        self.metrics = ["mean_squared_error"]
        # self.metrics = ["mean_absolute_error"]
        # self.metrics = ["mean_absolute_percentage_error"]
        # self.metrics = ["mean_squared_logarithmic_error"]

        self.data_header, self.data = csv_into_array(data_filename)
        self.label_header, self.label = csv_into_array(label_filename)

        self.data_length, self.data_size = self.data.shape
        self.label_length, self.label_size = self.label.shape

        if self.data.length != self.label.length:
            print("The data and label has different length.")
            exit(1)

        self.log_dir = str(time.time) + "_log"

        self.division_x = np.max(self.data)
        self.division_y = np.max(self.label)

        self.data /= self.division_x
        self.label /= self.division_y

        self.test_data = self.data[: int(self.data_length / 5)]
        self.train_data = self.data[int(self.data_length / 5):]

        self.test_label = self.label[:int(self.label_length/ 5)]
        self.train_label = self.label[int(self.label_length / 5):]

        self.tensorboard = keras.callbacks.TensorBoard(log_dir=self.log_dir + "/{}".format(time.time()))

        with tf.device(self.device):
            self.Graph = self.graph(True, self.log_dir)

        self.Graph.fit(self.train_data, self.train_label,
                       epoch=self.epoch,
                       batch_size=self.batch_size,
                       callbacks=[self.tensorboard],
                       validation_data=(self.test_data, self.test_label))

        self.test_loss, self.test_acc = self.Graph.evaluate(self.test_data, self.test_label)
        print("** test loss is : " + str(self.test_loss))
        print("** test acc is : " + str(self.test_acc))

        print("** Training Done **")
        self.equation = self.Graph.summary()
        print(self.equation)

        self.Graph.save(self.log_dir + "/saved_model.h5")

    def graph(self, reset, logdir):
        if reset:
            model = tf.keras.Sequential()
            model.add(layers.Dense(60, activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(60, activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(30, activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(20, activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(self.label_size, activation="sigmoid"))
            model.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics)
        else:
            model = keras.models.load_model(logdir + "/saved_model.h5")
        return model

    def volt_to_concentration(self, filename):
        header, data = csv_into_array(filename)
        result = self.Graph.predict(data, batch_size=self.batch_size)
        np.savetxt("converted_" + filename, result, delimiter=", ", header=str(header))
