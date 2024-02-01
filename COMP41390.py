#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import random
import numpy as np
import math
import os
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import copy

###########################################################################
# 1. Train an MLP with 2 inputs, 3-4+ hidden units and one output on the
# following examples (XOR function):
#    ((0, 0), 0)
#    ((0, 1), 1)
#    ((1, 0), 1)
#    ((1, 1), 0)
###########################################################################

# MLP object for first 2 problems
class MLP():
    def __init__(self, NI=2, NH=3, NO=1):
        self.NI = NI
        self.NH = NH
        self.NO = NO
        
        if NO > 1:
            print("This MLP class does not support multiple outputs. Try MLP 2 for that.")
            return
        
        # weights for the connections between each input and each hidden unit
        self.W1 = np.random.uniform(-0.1, 0.1, (self.NI, self.NH))
                
        # weights for the connections between each hidden unit and each output
        self.W2 = np.random.uniform(-0.1, 0.1, (self.NH, self.NO))
                
        # weight changes for the connections between each input and each hidden unit
        self.dW1 = np.zeros((self.NI, self.NH))
                
        # weight changes for the connections between each hidden unit and each output
        self.dW2 = np.zeros((self.NH, self.NO))
                
        # activations for layer between inputs and hidden units
        self.z1 = np.zeros((1,1,self.NH))
            
        # activations for layer between hidden units and outputs
        self.z2 = np.zeros((1,1,self.NO))
          
        # outputs for each hidden unit after its inputs*weights have been summed and sigmoided
        self.H = np.zeros((1,1,self.NH))
        
        # dict of outputs -- new entry for each example
        self.O = np.zeros((1,1,self.NO))

    def forward(self, inpt, epNo, egNo, trainingData):
        # method called with the current input vector, the epoch number, example number and
        # the dataset that the current input vector comes from
        
        # if the forward method is called for the first time
        # make sure that the ojects output, hidden unit output, outer layer activations and inner
        # layer activations are all the same dimension as the training data
        if len(trainingData) != len(self.O[0]):
            self.O = np.zeros((1,len(trainingData),self.NO))
            self.H = np.zeros((1,len(trainingData),self.NH))
            self.z2 = np.zeros((1,len(trainingData),self.NO))
            self.z1 = np.zeros((1,len(trainingData),self.NH))
        
        # if called with a new epoch, append another array of the same dimensions to each of 
        # the ojects output, hidden unit output, outer layer activations and inner layer activations
        # if the epoch number is a multiple of 100, this also adds another array to the object's output
        # this was for testing with a validation set to test the performance of the object along the way
        # over a large number of epochs – we may want to test the object on a validation set after every
        # 100 epochs out of 1000 to see if its performance is better before 1000 epochs of training.
        if (epNo+1) == len(self.O) or (epNo)%100 == 0:
            self.O = np.append(self.O, np.zeros((1,len(trainingData),self.NO)), axis=0)
            self.H = np.append(self.H, np.zeros((1,len(trainingData),self.NH)), axis=0)
            self.z2 = np.append(self.z2, np.zeros((1,len(trainingData),self.NO)), axis=0)
            self.z1 = np.append(self.H, np.zeros((1,len(trainingData),self.NH)), axis=0)
        
        # sigmoid function for each hidden unit
        def sigmoid(x):
            return 1.0/(1.0+np.exp(-x))
        
        # for each hidden unit
        for h in range(self.NH):  
            
            # the activation of this hidden unit for this epoch and example is
            # the dot product of the current example and the weights of the current hidden unit
            self.z1[epNo,egNo,h] = np.dot(inpt, self.W1[:, h])
            # the output of this hidden unit is then assigned the value of the sigmoid function
            # carried out on the activation described before
            self.H[epNo,egNo,h] = sigmoid(self.z1[epNo,egNo,h])
        
        # the outer layer activation is then defined as the dot product of each hidden unit's
        # output and the weights for the outer layer
        activation = np.dot(self.H[epNo,egNo], self.W2)
        self.z2[epNo, egNo] = activation
        
        # the output is defined as the sigmoid function carried out on the activation function
        # for the outer layer
        self.O[epNo, egNo] = sigmoid(activation)
        return
    
    def backward(self, inpt, epNo, egNo, target):
        # method called with the current input, epoch number, example number and target output
        # for the current input
        
        # sigmoid actiavation function for hidden and output units
        def sigmoid(x):
            return 1.0/(1.0+np.exp(-x))
        
        # sigmoid prime or sigmoid derivative function for calculating weight changes
        def sigmoid_derivative(x):
            return sigmoid(x) * (1 - sigmoid(x))
        
        # assign tp the output variable the value of the forward method for this epoch and example number
        output = self.O[epNo,egNo,0]
        
        # define the error as the output minus the target output
        error = (output-target)
        
        # outer layer
        # for each hidden unit
        for h in range(self.NH):
            
            # the weight change between the current hidden unit and the output is defined as
            # the error * the sigmoid_derivative function carried out on the outer layer activation
            # for the current epoch and example * the output of the current hidden unit
            weightChange = error*(sigmoid_derivative(self.z2[epNo,egNo]))*self.H[epNo,egNo,h]
            # add the weight change to the current value for this hidden unit in the outer layer
            # weight changes array
            self.dW2[h] += np.array(weightChange)
            
            # for each input unit
            for i in range(self.NI):
                # xi is defined as the value of input at the same index as the current input unit
                xi = inpt[i]
                
                # the weight change between each input unit and each hidden units is defined as
                # error * the sigmoid_derivative function carried out on the outer layer activation
                # for the current epoch and example * the weight between the current hidden unit
                # and the output * the sigmoid derivative function carried out on the activation
                # of the current hidden unit for the current epoch and example
                weightChange = error*(sigmoid_derivative(self.z2[epNo,egNo]))*self.W2[h][0]*\
sigmoid_derivative(self.z1[epNo,egNo,h])*xi
                
                # add the weight change to the current value for this hidden unit and input unit in
                # the outer layer weight changes array
                self.dW1[i,h] += weightChange
        
        # return the squared error for this example
        return (error**2)
    
    def updateWeights(self, learningRate=1):
        self.W1 -= learningRate*self.dW1
        self.dW1 = np.zeros((self.NI,self.NH))
        self.W2 -= learningRate*self.dW2
        self.dW2 = np.zeros((self.NH,self.NO))
        return
    
# results for comparing different numbers of hidden units
mlp_1_results = {}
mlp_2_results = {}
mlp_3_results = {}
mlp_4_results = {}

learning_rates = [0.1,1,5,10,20]
examples = np.array([[0,1],[1,1],[1,0],[0,0]])
labels = [1,0,1,0]

# training 4 networks with 3,4,5 and 6 hidden units for 1000 epochs, storing results in 
outString = ""
if os.path.exists("Q1_error.txt"):
    os.remove("Q1_error.txt")


for LR in learning_rates:
    
    print(f"For LR_{LR}")

    mlp_1 = MLP(2,3,1)
    mlp_2 = MLP(2,4,1)
    mlp_3 = MLP(2,5,1)
    mlp_4 = MLP(2,6,1)

    outString = f"""\
*********************************** \nFor LR={LR} \n***********************************"""

    # for each epoch
    for i in range(1000):
        error1 = 0
        error2 = 0
        error3 = 0
        error4 = 0
        
        print(f"For epoch_{i}")

        for eg in range(len(examples)):

            mlp_1.forward(examples[eg], i, eg, examples)
            error1 += mlp_1.backward(examples[eg], i, eg, labels[eg])

            mlp_2.forward(examples[eg], i, eg, examples)
            error2 += mlp_2.backward(examples[eg], i, eg, labels[eg])

            mlp_3.forward(examples[eg], i, eg, examples)
            error3 += mlp_3.backward(examples[eg], i, eg, labels[eg])

            mlp_4.forward(examples[eg], i, eg, examples)
            error4 += mlp_4.backward(examples[eg], i, eg, labels[eg])
            
        mlp_1.updateWeights(LR)
        mlp_2.updateWeights(LR)
        mlp_3.updateWeights(LR)
        mlp_4.updateWeights(LR)

        mlp_1_results[f"LR={LR} epoch_{i}"] = (error1/len(examples))
        mlp_2_results[f"LR={LR} epoch_{i}"] = (error2/len(examples))
        mlp_3_results[f"LR={LR} epoch_{i}"] = (error3/len(examples))
        mlp_4_results[f"LR={LR} epoch_{i}"] = (error4/len(examples))
        
        
        outString += f"""\n\n***********************************
3 hidden units -- ERROR at epoch_{i}: {error1/len(examples)}
4 hidden units -- ERROR at epoch_{i}: {error2/len(examples)}
5 hidden units -- ERROR at epoch_{i}: {error3/len(examples)}
6 hidden units -- ERROR at epoch_{i}: {error4/len(examples)}
***********************************"""

    with open("Q1_error.txt", 'a') as fh:
        fh.write(outString)

mlps = [mlp_1_results, mlp_2_results, mlp_3_results, mlp_4_results]
for i in mlps:
    j = mlps.index(i)
    min_error = min(list(i.values()))
    index = np.argmin(np.array(list(i.values())))
    index_name = list(i.keys())[index]
    print(f"For mlp_{j}, min error is: {min_error} at index_{index} -- {index_name}")

mlp = MLP(2,6,1)
epochs = 1000
LR = 20
examples = np.array([[0,1],[1,1],[1,0],[0,0]])
labels = [1,0,1,0]
outString = "Training MLP with 2 inputs, 6 hidden units and 1 output unit.\n"

if os.path.exists("Q1_training.txt"):
    os.remove("Q1_training.txt")

# train MLP
for i in range(epochs):
    error = 0
    for e in range(len(examples)):
        mlp.forward(examples[e],i,e,examples)
        error += mlp.backward(examples[e],i,e,labels[e])
    mlp.updateWeights(LR)
    outString += f"""\n\n***********************************
MEAN ERROR at epoch_{i}: {error/len(examples)}
***********************************"""

with open("Q1_training.txt", "w") as fh:
    fh.write(outString)


# 2. At the end of training, check if the MLP predicts correctly all
# the examples.
    
examples = np.array([[1,1],[1,0],[0,0],[0,1]])
labels = [0,1,0,1]
results = []

if os.path.exists("Q1_testing.txt"):
    os.remove("Q1_testing.txt")

for e in range(len(examples)):
    mlp.forward(examples[e],500,e,examples)
    if mlp.O[500,e] >=0.5:
        results.append(1)
    else:
        results.append(0)

correct_total = 0
for e in range(len(examples)):
    if results[e] == labels[e]:
        correct_total += 1
outString = f"""\
MLP with 8 hidden units, a learning rate of 0.1 trained for 1000 epochs \
predicted {(correct_total/len(examples))*100}% of the test set correctly.
"""

with open("Q1_testing.txt", "w") as fh:
    fh.write(outString)

###########################################################################

# 3. Generate 500 vectors containing 4 components each. The value of each component
# should be a random number between -1 and 1. These will be your input vectors. The
# corresponding output for each vector should be the sin() of a combination of the
# components. Specifically, for inputs: [x1 x2 x3 x4] the (single component) output
# should be: sin(x1-x2+x3-x4)
    
examples = np.random.uniform(-1,1,(500, 4))

labels = []
for i in range(500):
    x1=examples[i,0]
    x2=examples[i,1]
    x3=examples[i,2]
    x4=examples[i,3]
    labels.append(math.sin(x1-x2+x3-x4))
    
# Now train an MLP with 4 inputs, at least 5 hidden units and one output on 400 of
# these examples and keep the remaining 100 for testing.
    
X_train, X_test, y_train, y_test = train_test_split(examples, labels, test_size=0.2, random_state=23)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=23)

mlp_sto_1_results = {}
mlp_sto_2_results = {}
mlp_sto_3_results = {}
mlp_sto_4_results = {}
mlp_sto_5_results = {}

learning_rates = [0.1,1,5,10,20]
batchSizes = [32,64]
allData = list(zip(X_train,y_train))

outString = ""

if os.path.exists("Q2_stochastic_error.txt"):
    os.remove("Q2_stochastic_error.txt")

# for each learning rate    
for LR in learning_rates:
    # for each batch size
    for batchSize in batchSizes:
        
        print(f"Learning_rate_{LR} and batchSize_{batchSize}")
        
        # create an MLP object with 5,6,7,8 and 10 hidden units.
        mlp_sto_1 = MLP(4,5,1)
        mlp_sto_2 = MLP(4,6,1)
        mlp_sto_3 = MLP(4,7,1)
        mlp_sto_4 = MLP(4,8,1)
        mlp_sto_5 = MLP(4,10,1)

        outString = f"""\
\n*********************************** \nFor LR={LR} and batchSize = {batchSize} \
\n***********************************\n"""
        
        # for each epoch
        for i in range(1000):
            print(f"Epoch_{i}")
            
            # shuffle the training data
            random.shuffle(allData)
            X_train_shuffled,y_train_shuffled = zip(*allData)
            
            # create a batch and an array of the lables for each batch
            batch = np.array(X_train_shuffled[0:batchSize])
            batchLabels = np.array(y_train_shuffled[0:batchSize])
            
            # set the error for each MLP objct to 0
            error1,error2,error3,error4,error5 = 0,0,0,0,0
            
            # if the epoch is NOT a multiple of 1000
            if (i+1) % 1000 != 0:
            
                # for each example in the batch
                for eg in range(batchSize):
                    
                    # initialise the example as var batchEG and the label as batchLabel
                    batchEG = batch[eg]
                    batchLabel = batchLabels[eg]
                    
                    # run each MLP object on each example in the batch
                    # calculate the error for that example
                    # update the weights after all examples in this batch
                    mlp_sto_1.forward(batchEG, i, eg, batch)
                    error1 += mlp_sto_1.backward(batchEG, i, eg, batchLabel)

                    mlp_sto_2.forward(batchEG, i, eg, batch)
                    error2 += mlp_sto_2.backward(batchEG, i, eg, batchLabel)

                    mlp_sto_3.forward(batchEG, i, eg, batch)
                    error3 += mlp_sto_3.backward(batchEG, i, eg, batchLabel)
                
                    mlp_sto_4.forward(batchEG, i, eg, batch)
                    error4 += mlp_sto_4.backward(batchEG, i, eg, batchLabel)

                    mlp_sto_5.forward(batchEG, i, eg, batch)
                    error5 += mlp_sto_5.backward(batchEG, i, eg, batchLabel)
                    
                mlp_sto_1.updateWeights(LR)
                mlp_sto_2.updateWeights(LR)
                mlp_sto_3.updateWeights(LR)
                mlp_sto_4.updateWeights(LR)
                mlp_sto_5.updateWeights(LR)
                
                # for each MLP object make a string with the average error for the batch as well as 
                # the learning rate and batch size at each epoch
                outString += f"""\n***********************************
5HU LR_{LR} batchSize_{batchSize} -- ERROR at epoch_{i}: {error1/batchSize}
6HU LR_{LR} batchSize_{batchSize} -- ERROR at epoch_{i}: {error2/batchSize}
7HU LR_{LR} batchSize_{batchSize} -- ERROR at epoch_{i}: {error3/batchSize}
8HU LR_{LR} batchSize_{batchSize} -- ERROR at epoch_{i}: {error4/batchSize}
10HU LR_{LR} batchSize_{batchSize} -- ERROR at epoch_{i}: {error5/batchSize}
***********************************"""
            
            # if the epoch is a multiple of 1000
            else:
                
                # for each MLP object, create a new test object with the same number of hidden units
                # assign the weights of the test object to be the same as the ones for its corresponding
                # MLP object
                
                mlp_sto_1_test = MLP(4,5,1)
                mlp_sto_1_test.W1 = mlp_sto_1.W1
                mlp_sto_1_test.W2 = mlp_sto_1.W2
                
                mlp_sto_2_test = MLP(4,6,1)
                mlp_sto_2_test.W1 = mlp_sto_2.W1
                mlp_sto_2_test.W2 = mlp_sto_2.W2
                
                mlp_sto_3_test = MLP(4,7,1)
                mlp_sto_3_test.W1 = mlp_sto_3.W1
                mlp_sto_3_test.W2 = mlp_sto_3.W2
                
                mlp_sto_4_test = MLP(4,8,1)
                mlp_sto_4_test.W1 = mlp_sto_4.W1
                mlp_sto_4_test.W2 = mlp_sto_4.W2
                
                mlp_sto_5_test = MLP(4,10,1)
                mlp_sto_5_test.W1 = mlp_sto_5.W1
                mlp_sto_5_test.W2 = mlp_sto_5.W2
                
                # for each example in the validation set
                for j in range(len(X_val)):
                    # run the forward cycle with the MLP test object
                    # add the result of the backwards cycle to the error for each object
                    mlp_sto_1_test.forward(X_val[j], 0, j, X_val)
                    error1 += mlp_sto_1_test.backward(X_val[j], 0, j, y_val[j])
                    
                    mlp_sto_2_test.forward(X_val[j], 0, j, X_val)
                    error2 += mlp_sto_2_test.backward(X_val[j], 0, j, y_val[j])
                    
                    mlp_sto_3_test.forward(X_val[j], 0, j, X_val)
                    error3 += mlp_sto_3_test.backward(X_val[j], 0, j, y_val[j])
                    
                    mlp_sto_4_test.forward(X_val[j], 0, j, X_val)
                    error4 += mlp_sto_4_test.backward(X_val[j], 0, j, y_val[j])
                    
                    mlp_sto_5_test.forward(X_val[j], 0, j, X_val)
                    error5 += mlp_sto_5_test.backward(X_val[j], 0, j, y_val[j])
                
                # when each example in the test set has been worked
                # add the average error of each test object to its results dictionary
                # where the key is the learning rate, batch size and the number of training epochs
                mlp_sto_1_results[f"LR_{LR} batchsize_{batchSize} {i}_epochs"] = error1/len(X_val)
                mlp_sto_2_results[f"LR_{LR} batchsize_{batchSize} {i}_epochs"] = error2/len(X_val)
                mlp_sto_3_results[f"LR_{LR} batchsize_{batchSize} {i}_epochs"] = error3/len(X_val)
                mlp_sto_4_results[f"LR_{LR} batchsize_{batchSize} {i}_epochs"] = error4/len(X_val)
                mlp_sto_5_results[f"LR_{LR} batchsize_{batchSize} {i}_epochs"] = error5/len(X_val)
                
                # for each MLP test object make a string with the average error for the validation set as well as 
                # the learning rate and batch size after each 100th epoch
                # make sure the string has VALIDATION RESULTS in it to distinguish it from the training error
                # strings
                outString += f"""\n***********************************
VALIDATION RESULTS\n***********************************
5HU LR_{LR} batchSize_{batchSize} {i}_epochs: {error1/len(X_val)}
6HU LR_{LR} batchSize_{batchSize} {i}_epochs: {error2/len(X_val)}
7HU LR_{LR} batchSize_{batchSize} {i}_epochs: {error3/len(X_val)}
8HU LR_{LR} batchSize_{batchSize} {i}_epochs: {error4/len(X_val)}
10HU LR_{LR} batchSize_{batchSize} {i}_epochs: {error5/len(X_val)}
***********************************"""
        
        # when either the training or validation cycle has been completed, write the relevant string to the file
        # "Q2_stochastic_error.txt" and empty the string after
        with open("Q2_stochastic_error.txt", 'a') as fh:
            fh.write(outString)
        outString = ""
        
# Read the validation results to see what combination of parameters works the best
with open("Q2_stochastic_error.txt", "r") as fh:
    file = fh.read()
    
    
for i in range(len(file)):
    if file[i] == "\n":
        j = i+1
        if file[j] == "V":
            print(file[j:j+366])
            
#From the above code, we can see that batchSize 32 and learning rates of 0.1
# and 1 work best after 1000 epochs, so we will try these for 5000 epochs to
# see what MLP object performs best after that
            
learning_rates = [0.1,1]
batchSizes = [32]
allData = list(zip(X_train,y_train))

outString = ""

if os.path.exists("Q2_stochastic_error_refined.txt"):
    os.remove("Q2_stochastic_error_refined.txt")

# for each learning rate    
for LR in learning_rates:
    # for each batch size
    for batchSize in batchSizes:
        
        print(f"Learning_rate_{LR} and batchSize_{batchSize}")
        
        # create an MLP object with 5,6,7,8 and 10 hidden units.
        mlp_sto_1 = MLP(4,5,1)
        mlp_sto_2 = MLP(4,6,1)
        mlp_sto_3 = MLP(4,7,1)
        mlp_sto_4 = MLP(4,8,1)
        mlp_sto_5 = MLP(4,10,1)

        outString = f"""\
\n*********************************** \nFor LR={LR} and batchSize = {batchSize} \
\n***********************************\n"""
        
        # for each epoch
        for i in range(5000):
            print(f"Epoch_{i}")
            
            # shuffle the training data
            random.shuffle(allData)
            X_train_shuffled,y_train_shuffled = zip(*allData)
            
            # create a batch and an array of the lables for each batch
            batch = np.array(X_train_shuffled[0:batchSize])
            batchLabels = np.array(y_train_shuffled[0:batchSize])
            
            # set the error for each MLP objct to 0
            error1,error2,error3,error4,error5 = 0,0,0,0,0
            
            # if the epoch is NOT a multiple of 1000
            if (i+1) % 1000 != 0:
            
                # for each example in the batch
                for eg in range(batchSize):
                    
                    # initialise the example as var batchEG and the label as batchLabel
                    batchEG = batch[eg]
                    batchLabel = batchLabels[eg]
                    
                    # run each MLP object on each example in the batch
                    # calculate the error for that example
                    # update the weights after all examples in this batch
                    mlp_sto_1.forward(batchEG, i, eg, batch)
                    error1 += mlp_sto_1.backward(batchEG, i, eg, batchLabel)

                    mlp_sto_2.forward(batchEG, i, eg, batch)
                    error2 += mlp_sto_2.backward(batchEG, i, eg, batchLabel)

                    mlp_sto_3.forward(batchEG, i, eg, batch)
                    error3 += mlp_sto_3.backward(batchEG, i, eg, batchLabel)
                
                    mlp_sto_4.forward(batchEG, i, eg, batch)
                    error4 += mlp_sto_4.backward(batchEG, i, eg, batchLabel)

                    mlp_sto_5.forward(batchEG, i, eg, batch)
                    error5 += mlp_sto_5.backward(batchEG, i, eg, batchLabel)
                    
                mlp_sto_1.updateWeights(LR)
                mlp_sto_2.updateWeights(LR)
                mlp_sto_3.updateWeights(LR)
                mlp_sto_4.updateWeights(LR)
                mlp_sto_5.updateWeights(LR)
                
                # for each MLP object make a string with the average error for the batch as well as 
                # the learning rate and batch size at each epoch
                outString += f"""\n***********************************
5HU LR_{LR} batchSize_{batchSize} -- ERROR at epoch_{i}: {error1/batchSize}
6HU LR_{LR} batchSize_{batchSize} -- ERROR at epoch_{i}: {error2/batchSize}
7HU LR_{LR} batchSize_{batchSize} -- ERROR at epoch_{i}: {error3/batchSize}
8HU LR_{LR} batchSize_{batchSize} -- ERROR at epoch_{i}: {error4/batchSize}
10HU LR_{LR} batchSize_{batchSize} -- ERROR at epoch_{i}: {error5/batchSize}
***********************************"""
            
            # if the epoch is a multiple of 1000
            else:
                
                # for each MLP object, create a new test object with the same number of hidden units
                # assign the weights of the test object to be the same as the ones for its corresponding
                # MLP object
                
                mlp_sto_1_test = MLP(4,5,1)
                mlp_sto_1_test.W1 = mlp_sto_1.W1
                mlp_sto_1_test.W2 = mlp_sto_1.W2
                
                mlp_sto_2_test = MLP(4,6,1)
                mlp_sto_2_test.W1 = mlp_sto_2.W1
                mlp_sto_2_test.W2 = mlp_sto_2.W2
                
                mlp_sto_3_test = MLP(4,7,1)
                mlp_sto_3_test.W1 = mlp_sto_3.W1
                mlp_sto_3_test.W2 = mlp_sto_3.W2
                
                mlp_sto_4_test = MLP(4,8,1)
                mlp_sto_4_test.W1 = mlp_sto_4.W1
                mlp_sto_4_test.W2 = mlp_sto_4.W2
                
                mlp_sto_5_test = MLP(4,10,1)
                mlp_sto_5_test.W1 = mlp_sto_5.W1
                mlp_sto_5_test.W2 = mlp_sto_5.W2
                
                # for each example in the validation set
                for j in range(len(X_val)):
                    # run the forward cycle with the MLP test object
                    # add the result of the backwards cycle to the error for each object
                    mlp_sto_1_test.forward(X_val[j], 0, j, X_val)
                    error1 += mlp_sto_1_test.backward(X_val[j], 0, j, y_val[j])
                    
                    mlp_sto_2_test.forward(X_val[j], 0, j, X_val)
                    error2 += mlp_sto_2_test.backward(X_val[j], 0, j, y_val[j])
                    
                    mlp_sto_3_test.forward(X_val[j], 0, j, X_val)
                    error3 += mlp_sto_3_test.backward(X_val[j], 0, j, y_val[j])
                    
                    mlp_sto_4_test.forward(X_val[j], 0, j, X_val)
                    error4 += mlp_sto_4_test.backward(X_val[j], 0, j, y_val[j])
                    
                    mlp_sto_5_test.forward(X_val[j], 0, j, X_val)
                    error5 += mlp_sto_5_test.backward(X_val[j], 0, j, y_val[j])
                
                # when each example in the test set has been worked
                # add the average error of each test object to its results dictionary
                # where the key is the learning rate, batch size and the number of training epochs
                mlp_sto_1_results[f"LR_{LR} batchsize_{batchSize} {i}_epochs"] = error1/len(X_val)
                mlp_sto_2_results[f"LR_{LR} batchsize_{batchSize} {i}_epochs"] = error2/len(X_val)
                mlp_sto_3_results[f"LR_{LR} batchsize_{batchSize} {i}_epochs"] = error3/len(X_val)
                mlp_sto_4_results[f"LR_{LR} batchsize_{batchSize} {i}_epochs"] = error4/len(X_val)
                mlp_sto_5_results[f"LR_{LR} batchsize_{batchSize} {i}_epochs"] = error5/len(X_val)
                
                # for each MLP test object make a string with the average error for the validation set as well as 
                # the learning rate and batch size after each 100th epoch
                # make sure the string has VALIDATION RESULTS in it to distinguish it from the training error
                # strings
                outString += f"""\n***********************************
VALIDATION RESULTS\n***********************************
5HU LR_{LR} batchSize_{batchSize} {i}_epochs: {error1/len(X_val)}
6HU LR_{LR} batchSize_{batchSize} {i}_epochs: {error2/len(X_val)}
7HU LR_{LR} batchSize_{batchSize} {i}_epochs: {error3/len(X_val)}
8HU LR_{LR} batchSize_{batchSize} {i}_epochs: {error4/len(X_val)}
10HU LR_{LR} batchSize_{batchSize} {i}_epochs: {error5/len(X_val)}
***********************************"""
        
        # when either the training or validation cycle has been completed, write the relevant string to the file
        # "Q2_stochastic_error.txt" and empty the string after
        with open("Q2_stochastic_error_refined.txt", 'a') as fh:
            fh.write(outString)
        outString = ""

# Read the validation new results to see what combination of parameters works the best

with open("Q2_stochastic_error_refined.txt", "r") as fh:
    file = fh.read()

outString = ""
    
for i in range(len(file)):
    if file[i] == "\n":
        j = i+1
        if file[j] == "V":
            outString += "\n"+file[j:j+366]
            
# write this to a file
with open("Q2_stochastic_error_validation.txt", "w") as fh:
    fh.write(outString)
    
# As we can see, the MLP with 8 hidden units after 4999 epochs performs the
# best. We will now test this MLP object on the test set

if os.path.exists("Q2_testing.txt"):
    os.remove("Q2_testing.txt")

mlp_sto_4_test = MLP(4,8,1)
mlp_sto_4_test.W1 = mlp_sto_4.W1
mlp_sto_4_test.W2 = mlp_sto_4.W2

testError = 0
for i in range(len(X_test)):
    mlp_sto_4_test.forward(X_test[i],0,i,X_test)
    testError += mlp_sto_4_test.backward(X_test[i],0,i,y_test[i])
    
outString = f"""\
MLP with 8 hidden units, a learning rate of 0.1 and a batchSize of 32 trained for 1000 epochs \
predicted the outcomes with an error of {round((testError/len(X_test))*100, 4)}%.\
"""
    
with open("Q2_testing.txt", "w") as fh:
    fh.write(outString)
    
###########################################################################  

# Special test
# Train an MLP on the letter recognition set available in the UCI Machine Learning repository

# Split the dataset in a training part containing approximately 4/5 of the records,
# and a testing part containing the rest.

# Your MLP should have as many inputs as there are attributes (17), as many hidden
# units as you want (I suggest to start at ~10) and 26 outputs (one for each letter
# of the alphabet).
# You should train your MLP for at least 1000 epochs. After training, check how well
# you can classify the data reserved for testing.
    
# new class for multiple outputs

class MLP2():
    def __init__(self, NI=2, NH=3, NO=1):
        self.NI = NI
        self.NH = NH
        self.NO = NO
        
        # weights for the connections between each input and each hidden unit
        self.W1 = np.random.uniform(-0.1, 0.1, (self.NI, self.NH))
                
        # weights for the connections between each hidden unit and each output
        self.W2 = np.random.uniform(-0.1, 0.1, (self.NH, self.NO))
                
        # weight changes for the connections between each input and each hidden unit
        self.dW1 = np.zeros((self.NI, self.NH))
                
        # weight changes for the connections between each hidden unit and each output
        self.dW2 = np.zeros((self.NH, self.NO))
                
        # activations for layer between inputs and hidden unit
        self.z1 = np.zeros((1,1,self.NH))
            
        # activations for layer between hidden unit and outputs
        self.z2 = np.zeros((1,1,self.NO))
          
        # outputs for each hidden unit after its inputs*weights have been summed and sigmoided
        self.H = np.zeros((1,1,self.NH))
        
        # array of outputs -- new entry for each example
        self.O = np.zeros((1,1,self.NO))

    def forward(self, inpt, epNo, egNo, trainingData):
        # method called with the current input vector, the epoch number, example number and
        # the dataset that the current input vector comes from
        
        # if the forward method is called for the first time
        # make sure that the ojects output, hidden unit output, outer layer activations and inner
        # layer activations are all the same dimension as the training data
        
        if len(self.O[0]) != len(trainingData):
            self.O = np.zeros((1,len(trainingData),self.NO))
            self.H = np.zeros((1,len(trainingData),self.NH))
            self.z2 = np.zeros((1, len(trainingData),self.NO))
            self.z1 = np.zeros((1,len(trainingData),self.NH))
        
        # if called with a new epoch, append another array of the same dimensions to each of 
        # the ojects output, hidden unit output, outer layer activations and inner layer activations
        # if the epoch number is a multiple of 100, this also adds another array to the object's output
        # this was for testing with a validation set to test the performance of the object along the way
        # over a large number of epochs – we may want to test the object on a validation set after every
        # 100 epochs out of 1000 to see if its performance is better before 1000 epochs of training.
        if (epNo+1) == len(self.O) or epNo%250 == 0:
            self.O = np.append(self.O, np.zeros((1,len(trainingData),self.NO)), axis=0)
            self.H = np.append(self.H, np.zeros((1,len(trainingData),self.NH)), axis=0)
            self.z2 = np.append(self.z2, np.zeros((1,len(trainingData),self.NO)), axis=0)
            self.z1 = np.append(self.H, np.zeros((1,len(trainingData),self.NH)), axis=0)
        
        # sigmoid function for each hidden unit
        def sigmoid(x):
            return 1.0/(1.0+np.exp(-x))
        
        # for each hidden unit
        for h in range(self.NH):
            # the activation of this hidden unit for this epoch and example is
            # the dot product of the current example and the weights of the current hidden unit
            self.z1[epNo,egNo,h] = np.dot(inpt, self.W1[:, h])
            # the output of this hidden unit is then assigned the value of the sigmoid function
            # carried out on the activation described before
            self.H[epNo,egNo,h] = sigmoid(self.z1[epNo,egNo,h])
        
        # for each output unit
        for o in range(self.NO):
            # the outer layer activation is then defined as the dot product of each hidden unit's
            # output and the weights between each hidden unit and the current output unit
            activation = np.dot(self.H[epNo,egNo], self.W2[:,o])
            self.z2[epNo, egNo, o] = activation
        
            # the output for this output unit is then defined as the sigmoid function carried out
            # on the activation for the current output unit
            self.O[epNo, egNo, o] = sigmoid(activation)
        return
    
    def backward(self, inpt, epNo, egNo, targetVector):
        # method called with the current input, epoch number, example number and target output
        # for the current input
        
        # sigmoid actiavation function for hidden and output units
        def sigmoid(x):
            return 1.0/(1.0+np.exp(-x))
        
        # sigmoid prime or sigmoid derivative function for calculating weight changes
        def sigmoid_derivative(x):
            return sigmoid(x) * (1 - sigmoid(x))
        
        # will be returning the average error across all outputs, each of which will be calculated
        # individually, so initialise the final error as 0
        finalError = 0
        
        # outer layer
        # for each output unit
        for o in range(self.NO):
            # target is the target output for that unit
            target = targetVector[o]
            # output is the output for that unit
            output = self.O[epNo,egNo,o]
            # error is the output minus the target for this output unit
            error = (output-target)
            # add this error to the finalError variable
            finalError += (output-target)
            
            # for each hidden unit and the current output unit
            for h in range(self.NH):
                # the weight change between the current hidden unit and the current output unit is defined as
                # the error * the sigmoid_derivative function carried out on the outer layer activation
                # for the current epoch and example * the output of the current hidden unit
                weightChange = error*(sigmoid_derivative(self.z2[epNo,egNo,o]))*self.H[epNo,egNo,h]
                # add the weight change to the current value for this hidden unit/output unit in the outer layer
                # weight changes array
                self.dW2[h,o] += np.array(weightChange)
                
                # for each input unit and the current hidden unit and output unit
                for i in range(self.NI):
                    # xi is defined as the value of input at the same index as the current input unit
                    xi = inpt[i]
                    # the weight change between each input unit and each hidden units is defined as
                    # error * the sigmoid_derivative function carried out on the outer layer activation
                    # for the current epoch and example * the weight between the current hidden unit
                    # and the output * the sigmoid derivative function carried out on the activation
                    # of the current hidden unit for the current epoch and example
                    weightChange = error*(sigmoid_derivative(self.z2[epNo,egNo,o]))\
                    *self.W2[h,o]*sigmoid_derivative(self.z1[epNo,egNo,h])*xi
                    # add the weight change to the current value for this hidden unit and input unit in
                    # the outer layer weight changes array
                    self.dW1[i,h] += weightChange
                    
        # return the squared error for this example
        return (finalError**2)
    
    def updateWeights(self, learningRate=1):
        self.W1 -= learningRate*self.dW1
        self.dW1 = np.zeros((self.NI,self.NH))
        self.W2 -= learningRate*self.dW2
        self.dW2 = np.zeros((self.NH,self.NO))
        return

# read and format UCI data    
df = pd.read_csv("letter-recognition.data", names=["letter", 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
labels = df.pop('letter')
normalized_df=(df-df.min())/(df.max()-df.min())
X = normalized_df.to_numpy()
y = np.zeros((20000,26))
for i in range(20000):
    letter = labels[i]
    letterIndex = string.ascii_uppercase.index(letter)
    y[i,letterIndex]=1
    
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=23)

# learning rate of 0.1 and batchSize had best results, but these were were varied in training
# the number of hidden units tested was also varied
learning_rates = [0.1]
batchSizes = [128]
allData = list(zip(X_train,y_train))

# for each learning rate  
for LR in learning_rates:
    # for each batch size
    for batchSize in batchSizes:
        
        print(f"Learning_rate_{LR} and batchSize_{batchSize}")
        
        # create an MLP object with 10 hidden units, for this example.
        mlp_sto_1 = MLP2(16,10,26)
        
        # for each epoch
        for i in range(1000):
            print(f"Epoch_{i}")
            
            # shuffle the training data
            random.shuffle(allData)
            X_train_shuffled,y_train_shuffled = zip(*allData)
            
            # create a batch and an array of the lables for each batch
            batch = np.array(X_train_shuffled[0:batchSize])
            batchLabels = np.array(y_train_shuffled[0:batchSize])
            
            # set the error for each MLP objct to 0
            error1 = 0
                
                # for each example in the batch
            for eg in range(batchSize):

                # initialise the example as var batchEG and the label as batchLabel
                batchEG = batch[eg]
                batchLabel = batchLabels[eg]

                # run each MLP object on each example in the batch
                # calculate the error for that example
                # update the weights after all examples
                mlp_sto_1.forward(batchEG, i, eg, batch)
                error1 += mlp_sto_1.backward(batchEG, i, eg, batchLabel)
            mlp_sto_1.updateWeights(LR)
            
# tracking results
finalTotal = 0
testResults = []
additions = [0,500,1000,1500,2000,2500,3000,3500]

for addition in additions:
    mlp_sto_1_test = MLP2(16,10,26)
    mlp_sto_1_test.W1 = mlp_sto_1_3000_epochs.W1
    mlp_sto_1_test.W2 = mlp_sto_1_3000_epochs.W2

    for i in range(len(X_test[0:500])):
        print(i+addition)
        mlp_sto_1_test.forward(X_test[(i+addition)], 0, (i+addition), X_test)
        
    total = 0
    for i in range(500):
        output_max_index = mlp_sto_1_test.O[0,(i+addition)].argmax()
        desired_max_index = y_test[(i+addition)].argmax()
        if output_max_index == desired_max_index:
            total+=1
    total /= 500
    print(total)
    finalTotal += total
    testResults.append(total*100)

    outString = f"""\
    \n\nMLP with 10 hidden units, a learning rate of {LR} and a batchSize of {batchSize} trained for 3000 epochs \
    predicted from testExample {addition} to {addition+499} {round(total*100, 4)}% correctly.\
    """

    with open("Q3_test_error_10HU_LR_0_1_BS_128_3000_epochs.txt","a") as fh:
        fh.write(outString)