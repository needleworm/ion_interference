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
    def __init__(self, function, slope, intercept):
        self.function = function
        self.slope = slope
        self.intercept = intercept

    def __str__(self):
        if self.function == "exp":
            return "Y = " + str(self.slope) + " exp(X) + " + str(self.intercept)

        elif self.function == "expexp":
            return "Y = " + str(np.exp(self.intercept)) + " exp( " + str(self.slope) + " X )"

    def calculate(self, x):
        if self.function == "exp":
            return self.slope * np.exp(x) + self.intercept
        elif self.function == "expexp":
            return np.exp(self.intercept) * np.exp(self.slope * x)


class Exp:
    def __init__(self, data_filename, label_filename):
        self.data_header, self.data = csv_into_array(data_filename)
        self.label_header, self.label = csv_into_array(label_filename)

        if self.data.shape != self.label.shape:
            print("The data and label has different shape.")
            exit(1)

        self.data_size, self.data_category = self.data.shape

        self.slopes = []
        self.intercepts = []

        self.regression(np.exp(self.data), self.label) # x는 exp 씌우고 y는 그대로
        self.equation = make_equation("exp", self.slopes, self.intercepts)

    def regression(self, data, label):
        for i in range(len(self.data_category)):
            regression_result = np.polyfit(data[:, i], label[:, i])
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

    def volt_to_concentration(self, filename):
        header, data = csv_into_array(filename)
        for i in range(len(self.equation)):
            data[:, i] = self.equation[i].calculate(data[:, i])
        np.savetxt("converted_" + filename, data, delimiter=", ", header=header)


class ExpExp:
    def __init__(self, data_filename, label_filename):
        self.data_header, self.data = csv_into_array(data_filename)
        self.label_header, self.label = csv_into_array(label_filename)

        if self.data.shape != self.label.shape:
            print("The data and label has different shape.")
            exit(1)

        self.data_size, self.data_category = self.data.shape

        self.slopes = []
        self.intercepts = []
        self.r_values = []
        self.p_values = []
        self.std_errs = []

        self.regression(np.exp(self.data), self.label) # x는 exp 씌우고 y는 로그 씌우고
        self.equation = make_equation("expexp", self.slopes, self.intercepts)

    def regression(self, data, label):
        for i in range(len(self.data_category)):
            regression_result = np.polyfit(data[:, i], label[:, i])
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

    def volt_to_concentration(self, filename):
        header, data = csv_into_array(filename)
        for i in range(len(self.equation)):
            data[:, i] = self.equation[i].calculate(data[:, i])
        np.savetxt("converted_" + filename, data, delimiter=", ", header=header)


class DeepLearning:
    def __init__(self, data_filename, label_filename):
        self.bathc_size = 128
        self.epoch = 10
        self.device = "/gpu:0"

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
        print(self.Graph.summary())

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
            model.add(layers.Dense(self.Y_size, activation="sigmoid"))
            model.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics)
        else:
            model = keras.models.load_model(logdir + "/saved_model.h5")
        return model

    def volt_to_concentration(self, filename):
        header, data = csv_into_array(filename)
        result = self.Graph.predict(data, batch_size=self.batch_size)
        np.savetxt("converted_" + filename, result, delimiter=", ", header=header)
