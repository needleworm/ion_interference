# ion_interference
Machine Learning Approach to Remove Ion Interference Effect

# Citation
[1] B. Ban, D. Ryu and M. Lee, "Machine Learning Approach to Remove Ion Interference Effect in Agricultural Nutrient Solutions," 2019 International Conference on Information and Communication Technology Convergence (ICTC), Jeju Island, Korea (South), 2019, pp. 1156-1161, doi: 10.1109/ICTC46691.2019.8939812.

# Overview
This repository provides a machine learning approach to remove ion interference effect on ISE(ion selective electrodes).
It's main purpose is to presents the readjustment function from citation [1].

![equation](https://latex.codecogs.com/gif.latex?C_%7Br%7D%20%3D%20%5Cmu%20%28TDS%29%20%5Ctimes%20C_%7BISE%7D)


![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20%28TDS%29%20%5Capprox%20%5Cfrac%7BC_%7Br%7D%7D%7BC_%7BISE%7D%7D)


# How to use
## Dependencies
> pip install numpy scipy tensorflow

## Linear model
> from Readjust import linear as L

> model = Q.model(filename)

> equation = Q.equation

> readjusted_value = Q.readjust(raw_value)


## Quadratic model from citation [1]
> from Readjust import quadratic as Q

> model = Q.model(filename)

> equation = Q.equation

> readjusted_value = Q.readjust(raw_value)

## Deep learning model from citation [2]
> from Readjust import linear as L

> model = Q.model(filename)

> equation = Q.equation

> readjusted_value = Q.readjust(raw_value)


