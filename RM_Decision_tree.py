import numpy as np
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]
X_train_2=X_train
def reduce_array(X):
    reduced_array=[]
    for i in range(0,len(X)):
        reduced_array.append([X[i][0],np.log10(X[i][1]+X[i][2]+1),X[i][3],np.log2(X[i][4]+X[i][5]+1)])
    reduced_array=np.asarray(reduced_array)
    return reduced_array
def normalize_array(X,Y):
    average_array=np.mean(Y,axis=0)
    std=np.std(Y,axis=0)
    for i in range(0,len(X)):
        for j in range(1,len(X[0])):
            X[i][j]=(X[i][j]-average_array[j])/std[j]
    X=np.asarray(X)
    return X
X_train=normalize_array(reduce_array(X_train),reduce_array(X_train))
X_test=normalize_array(reduce_array(X_test),reduce_array(X_test))

from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")
