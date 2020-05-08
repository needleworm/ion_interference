"""
  Author    Byunghyun Ban
  Email     halfbottle@sangsang.farm
  CTO of Imagination Garden Inc.
  http://sangsang.farm
"""
import numpy as np
from scipy import stats
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
            print("The data and label has different length.")
            exit(1)

        self.data_size, self.data_category = self.data.shape

        self.slopes = []
        self.intercepts = []
        self.r_values = []
        self.p_values = []
        self.std_errs = []

        self.regression(np.exp(self.data), self.label) # x는 exp 씌우고 y는 그대로
        self.equation = make_equation("exp", self.slopes, self.intercepts)

    def regression(self, data, label):
        for i in range(len(self.data_category)):
            slope, intercept, r_value, p_value, std_err = stats.linregress(data[:, i], label[:, i])
            self.slopes.append(slope)
            self.intercepts.append(intercept)
            self.r_values.append(r_value)
            self.p_values.append(p_value)
            self.std_errs.append(std_err)

        report = "Regression Result\n"
        for i in range(len(self.slopes)):
            report += "Regression number " + str(i + 1) + ".\n"
            report += "Slope : " + str(self.slopes[i]) + "\n"
            report += "Intercept : " + str(self.intercepts[i]) + "\n"
            report += "R Value : " + str(self.r_values[i]) + "\n"
            report += "P Value : " + str(self.p_values[i]) + "\n"
            report += "Std Error : " + str(self.std_errs[i]) + "\n\n\n"

        report_file = open("Report_" + str(time.time()) + ".txt", "w")
        report_file.write(report)
        report_file.close()
        print(report)

    def volt_to_concentration(self, filename):
        header, data = csv_into_array(filename)
        for i in range(len(self.equation)):
            data[:, i] = self.equation[i].calculate(data[:, i])
        np.savetxt("converted_" + filename, delimiter=", ", header=header)


class ExpExp:
    def __init__(self, data_filename, label_filename):
        self.data_header, self.data = csv_into_array(data_filename)
        self.label_header, self.label = csv_into_array(label_filename)

        if self.data.shape != self.label.shape:
            print("The data and label has different length.")
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
        for i in range(self.data_category):
            slope, intercept, r_value, p_value, std_err = stats.linregress(data[:, i], label[:, i])
            self.slopes.append(slope)
            self.intercepts.append(intercept)
            self.r_values.append(r_value)
            self.p_values.append(p_value)
            self.std_errs.append(std_err)

        report = "Regression Result\n"
        for i in range(len(self.slopes)):
            report += "Regression number " + str(i + 1) + ".\n"
            report += "Slope : " + str(self.slopes[i]) + "\n"
            report += "Intercept : " + str(self.intercepts[i]) + "\n"
            report += "R Value : " + str(self.r_values[i]) + "\n"
            report += "P Value : " + str(self.p_values[i]) + "\n"
            report += "Std Error : " + str(self.std_errs[i]) + "\n\n\n"

        report_file = open("Report_" + str(time.time()) + ".txt", "w")
        report_file.write(report)
        report_file.close()
        print(report)

    def volt_to_concentration(self, filename):
        header, data = csv_into_array(filename)
        for i in range(len(self.equation)):
            data[:, i] = self.equation[i].calculate(data[:, i])
        np.savetxt("converted_" + filename, delimiter=", ", header=header)


class DeepLearning:
    def __init__(self, data_filename, label_filename):
        self.data_header, self.data = csv_into_array(data_filename)
        self.label_header, self.label = csv_into_array(label_filename)

        if self.data.shape != self.label.shape:
            print("The data and label has different length.")
            exit(1)

    def readjust(self, raw_value):
        return

