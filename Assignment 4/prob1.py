import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine

cancer = load_breast_cancer()
iris = load_iris()
digits = load_digits()
wine = load_wine()

x_array= np.arange(0,20,1)
C_array=[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]

num_folds = 5
models_list1 = []
models_list2 = []
models_list3 = []
models_list4 = []

for c in C_array :
    models_list1.append(('LR(C='+str(c)+')', LogisticRegression(C=c)))
    models_list2.append(('P(C=' + str(c) + ')', Perceptron(alpha=c, penalty='l2')))
    models_list3.append(('SVM(C='+str(c)+')', LinearSVC(C=c)))

for x in x_array :
    models_list4.append(('KNN(x=' + str(x) + ')', KNeighborsClassifier(n_neighbors=(6 * x + 1))))

entire_models_list = [models_list1, models_list2, models_list3, models_list4]

#------------------------------Breast Cancer----------------------------------------------------

for models_list in entire_models_list :
    results = []
    names = []
    for name, model in models_list:
        kfold = KFold(n_splits=num_folds, random_state=123)
        cv_result = cross_val_score(model, cancer.data, cancer.target, cv=kfold, scoring='accuracy')
        results.append(cv_result)
        names.append(name)
        print("%s Accuracy score:  Mean - %f ;STD - (%f) " % (name, cv_result.mean(), cv_result.std()))

    fig = plt.figure()
    fig.suptitle('Performance Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names, fontsize=5)
    plt.show()

#---------------------------------------------Iris--------------------------------------------------

for models_list in entire_models_list :
    results = []
    names = []
    for name, model in models_list:
        kfold = KFold(n_splits=num_folds, random_state=123)
        cv_result = cross_val_score(model, iris.data, iris.target, cv=kfold, scoring='accuracy')
        results.append(cv_result)
        names.append(name)
        print("%s Accuracy score:  Mean - %f ;STD - (%f) " % (name, cv_result.mean(), cv_result.std()))

    fig = plt.figure()
    fig.suptitle('Performance Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names, fontsize=5)
    plt.show()

#---------------------------------------------Digits--------------------------------------------------

for models_list in entire_models_list :
    results = []
    names = []
    for name, model in models_list:
        kfold = KFold(n_splits=num_folds, random_state=123)
        cv_result = cross_val_score(model, digits.data, digits.target, cv=kfold, scoring='accuracy')
        results.append(cv_result)
        names.append(name)
        print("%s Accuracy score:  Mean - %f ;STD - (%f) " % (name, cv_result.mean(), cv_result.std()))

    fig = plt.figure()
    fig.suptitle('Performance Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names, fontsize=5)
    plt.show()

#---------------------------------------------Wine--------------------------------------------------

for models_list in entire_models_list :
    results = []
    names = []
    for name, model in models_list:
        kfold = KFold(n_splits=num_folds, random_state=123)
        cv_result = cross_val_score(model, wine.data, wine.target, cv=kfold, scoring='accuracy')
        results.append(cv_result)
        names.append(name)
        print("%s Accuracy score:  Mean - %f ;STD - (%f) " % (name, cv_result.mean(), cv_result.std()))

    fig = plt.figure()
    fig.suptitle('Performance Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names, fontsize=5)
    plt.show()

