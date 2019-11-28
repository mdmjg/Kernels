#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np

def read_data(text):
    X = []
    Y = []
    with open(text, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            X.append((lines[i].split(',')[1:]))
            Y.append((lines[i].split(',')[0]))
    # make labels ints and normalize X
    for i in range(len(Y)):
        Y[i] = int(Y[i])
        for j in range(len(X[0])):
            X[i][j] = (((2*(int(X[i][j])))/255)-1)

    return np.array(X), np.array(Y)


X_train, Y_train = read_data('mnist_train.txt')
X_test, Y_test = read_data('mnist_test.txt')



# In[7]:


# turn each label into 0s and 1s for one-versus all classification

def getBinaryClassifiers(data):
    total_labels = []
    for i in range(10):
        classifier = [None] * len(data)
        for j in range(len(data)):
            if data[j] != i: 
                classifier[j] = -1
            else:
                classifier[j] = 1
        total_labels.append(classifier)
    return total_labels

binary_classifiers = getBinaryClassifiers(Y_train)
print(len(binary_classifiers))


# Train 10 binary classifiers using pegasos with linear kernel


# In[11]:


from sklearn import svm
import random
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# Create a classifier: a support vector classifier
# use linear kernel to train the 10 binary classifiers
# kernel = svm.SVC(probability=False,  kernel="linear", C=2.8, gamma=.0073,verbose=10)




def k_pegasos(x, y, lamb, kernel, num_iter):
    num_instances = len(x[0])
    alpha = np.zeros(num_instances)
    for t in range(1, num_iter+1):
        step = 1.0/(t*lamb)
        i=random.randint(0,num_instances-1)
        result = 0
        for j in range(num_instances):
            result += (alpha[j]*y[i]*kernel[i][j])
        if( result < 1):
            alpha[i] = (1-step*lamb)*alpha[i] + step*y[i]
        else:
            alpha[i] = (1-step*lamb)*alpha[i] 
            
    return alpha




# compute the scores for each of the classifiers

def compute_scores(x, y, w):
    return np.dot(x.transpose(), w)


kernel = pairwise_kernels(X_train, metric='linear')
classifiers = getBinaryClassifiers(Y_train)
weights = []
for cls in classifiers:
    w = k_pegasos(X_train, cls, 0.01, kernel, 10)
    weights.append(w)

    
for x in X_train:
    scores = []
    max_score = 0
    best_w = 0
    for w in weights:
        score = compute_scores(x, Y_train, w)
        if score > max_score:
            max_score = score
            best_w = w
        scores.append(score)
print(max_score)


# In[13]:


# kfold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


X_total = np.concatenate((X_train,X_test), axis = 0)
Y_total = np.concatenate((Y_train,Y_test), axis = 0)


def compute_error(x, y, w):
    return mean_squared_error(y, np.dot(x, w))



#compute k fold cross validation
lambs = [2**-9,2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 1, 2]
k_fold = KFold(n_splits=5, shuffle=True)
k_fold.get_n_splits(X_total)
errors = []
for l in lambs:
    l_errors = []
    for train, test in k_fold.split(X_total):
        x_train, x_test = X_total[train], X_total[test]
        y_train, y_test = Y_total[train], Y_total[test]
    
        w = k_pegasos(x_train, y_train, l, kernel, 10)
        l_errors.append(compute_error(x_train, y_train, w))
    errors.append(np.mean(l_errors))
    
plt.figure(figsize = (20, 10))
plt.plot(lambs, errors)
plt.xlabel("Lambdas")
plt.ylabel("Errors")
        


# In[14]:


# normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
X_train2, Y_train2 = read_data('mnist_train.txt')
X_test2, Y_test2 = read_data('mnist_test.txt')


scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train2)
X_train2 = scaler.transform(X_train2)

scaler.fit(Y_train2.reshape(1, -1))
Y_train2 = scaler.transform(Y_train2.reshape(1, -1))
Y_train2 = Y_train2.transpose()

scaler.fit(X_test2)
X_test2 = scaler.transform(X_test2)

scaler.fit(Y_test2.reshape(1, -1))
Y_test2 = scaler.transform(Y_test2.reshape(1, -1))
Y_test2 = Y_test2.transpose()


classifier = OneVsRestClassifier(svm.SVC())

classifier.fit(X_train2, Y_train2)

pred = classifier.predict(X_test2)

mse = mean_squared_error(Y_test2, pred)

print("The test error is ", mse)


# In[17]:


from sklearn.model_selection import cross_val_score
ten_cross_score = cross_val_score(classifier,X_total,Y_total,cv=10)
print("The ten fold cross validation score is ", ten_cross_score)


# In[ ]:


Cs = [0.2, 0.4, 0.6, 0.8, 1.0, 2]
gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]

for c in Cs:
    for g in gammas:
        classifier = OneVsRestClassifier(svm.SVC(C=c, gamma = g))

        classifier.fit(X_train2, Y_train2)

        pred = classifier.predict(X_test2)

        mse = mean_squared_error(Y_test2, pred)
        ten_cross_score = cross_val_score(classifier,total_X,total_Y,cv=10)
        print("For c: ", c, "and gamma:  ", "test error is", mse, " and cross validation error is ", np.mean(ten_cross_score))

