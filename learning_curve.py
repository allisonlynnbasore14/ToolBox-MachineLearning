""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model():
    data = load_digits()
    #X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.4)
    num_trials = 10
    train_percentages = range(5, 95, 5)
    test_accuracies = numpy.zeros(len(train_percentages))
    

    
    for per in train_percentages: #running though all percents
        totalval = 0
        for i in range(num_trials): #running through each one 10 times
            X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=per) #running the tests
            model = LogisticRegression(C=10) #making the model
            model.fit(X_train, y_train) #fitting the model
            totalval = totalval + model.score(X_test,y_test) #collecting the output for each run and adding together to find average
        aver = totalval/num_trials #finding the average for each percent
        run = int(per/5 -1) #calculating the run totals, the /5 thing is get getting the percents to correspomd to the graph, so perecnt of 20/5 means it is on the 4th run through and the 4th index of value outputs
        test_accuracies[run] = aver #storinig the values to be plotted


    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()


if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    #display_digits()
    train_model()
