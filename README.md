# ion_interference
Machine Learning Approach to Remove Ion Interference Effect

# iEnvCmplx
Chemical Complex System Simulator for Ion Interaction Environment

# Citation
[1] B. Ban, D. Ryu and M. Lee, "[Machine Learning Approach to Remove Ion Interference Effect in Agricultural Nutrient Solutions](https://arxiv.org/abs/1907.10794)" 2019 International Conference on Information and Communication Technology Convergence (ICTC), Jeju Island, Korea (South), 2019, pp. 1156-1161, doi: 10.1109/ICTC46691.2019.8939812.

[2] Ban, Byunghyun. "Deep learning method to remove chemical, kinetic and electric artifacts on ISEs." 2020 International Conference on Information and Communication Technology Convergence (ICTC). IEEE, 2020.

Please cite both papers.

# Overview
This repository provides a machine learning approach to remove ion interference effect on ISE(ion selective electrodes).
It's main purpose is to presents the readjustment function from reference [1].

![equation](https://latex.codecogs.com/gif.latex?C_%7Br%7D%20%3D%20%5Cmu%20%28TDS%29%20%5Ctimes%20C_%7BISE%7D)

The scripts regress on the equation below:

![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20%28TDS%29%20%5Capprox%20%5Cfrac%7BC_%7Br%7D%7D%7BC_%7BISE%7D%7D)

Then the predicted function u performs ion-interference effect removal.

Also, it provides calibration tool for ion sensors.

# Input Data file format
 2 csv files. One contains X, the other contains Y. The first line is removed during parsing. Please do not write data here. The first row is prepared for the header.
 
 For calibration, X should be measured voltage and Y should be theoretical concentration.
 
 For ion interference effect removal, X should be experimental concentration and Y should be theoretical concentration.
 
 For refinement of data after training, please feed a one-column file, whose rows contain single value each.
 

# 1. Dependencies
> pip install numpy scipy tensorflow

# 2. Calibration
input : voltage

output : concentration

>from ion_preprocessing import calibration as IC

### (1) Exponential model from Theory
> cali_model = IC.Exp(data_filename, label_filename)

### (2) Double Exponential model from reference [1]
> cali_model = IC.ExpExp(data_filename, label_filename)

### (3) Deep Learning model from reference [2]
> cali_model = IC.DeepLearning(data_filename, label_filename)

### (4) Usage

> equation = cali_model.equation

> readjusted_value = cali_model.readjust(raw_value)

> cali_model.volt_to_concentration(volt_file_filename)

# 3. Remove Ion Interference Effect
input : concentration

output : concentration


>from ion_preprocessing import readjustment as IR

### (1) Linear model
> model = IR.Linear(data_filename, label_filename)

### (2) Quadratic model from reference [1]
> model = IR.Quadratic(data_filename, label_filename)

### (3) Deep learning model from citation [2]
> model = IR.DeepLearning(data_filename, label_filename)

### (4) Usage

> equation = model.equation

> readjusted_value = model.readjust(raw_value)

> model.refine_concentration(concentration_file_filename)

# 4 Comments on Deep learning
When you run the deep learning method, a log directory will appear on your working directory.
During training, the best result will be continuously saved in the log directory.

It runs 100 epoch of training at first. And during the training process, the module detects 'Which epoch showed the best test result'.

The the best fitting weights and conversion result is saved.

## To load weight
Load the class with weight name specified.

> model = IR.DeepLearning(data_filename, label_filename, weight_filename)
  
> model = IC.DeepLearning(data_filename, label_filename, weight_filename)


The weight used on the reference [2] is attached.
