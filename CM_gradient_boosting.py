import numpy as np
# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]
def reduce_array(X):
    reduced_array=[]
    for i in range(0,len(X)):
        to_append=[]
        for j in range(0,20):
            to_append.append((X[i][j]+X[i][j+1]+X[i][j+2])/3)
        for j in range(60,len(X[0])):
            to_append.append(X[i][j])
        reduced_array.append(to_append)
    reduced_array=np.asarray(reduced_array)
    return(reduced_array)
def average_array(X,power):
    average=[]
    for i in range(0,len(X[0])):
        to_append=0
        for j in range(1,len(X)):
            to_append=to_append+(X[j][i]**power)
        to_append=to_append/(len(X))
        to_append=to_append**(1/5)
        average.append(to_append)
    return average
def binary_array(X,average):
    X_binary=[]
    for i in range(0, len(X)):
        to_append=[]
        for j in range(0,len(X[0])):
            if(X[i][j]<=average[j]):
                to_append.append(0)
            else:
                to_append.append(1)
        X_binary.append(to_append)
    return X_binary

train_reduce=reduce_array(X_train)
test_reduce=reduce_array(X_test)
X_train_binary=binary_array(X_train,average_array(X_train,5))
X_test_binary=binary_array(X_test,average_array(X_test,5))
train_reduce_binary=binary_array(train_reduce,average_array(train_reduce,5))
test_reduce_binary=binary_array(test_reduce,average_array(test_reduce,5))
train_binary_reduce=reduce_array(X_train_binary)
test_binary_reduce=reduce_array(X_test_binary)

from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
model.fit(X_train_binary, y_train)
y_pred=model.predict(X_test_binary)
test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('GradientBoostingClassifier.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")
