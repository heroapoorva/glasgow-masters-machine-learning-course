import numpy as np
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]
y_train=np.log2(y_train)
def reduce_array(X):
    reduced_array=[]
    for i in range(0,len(X)):
        reduced_array.append([X[i][0],np.log2(((X[i][1]+X[i][2])/1000)+2),X[i][3],np.log2(X[i][4]+X[i][5]+2)])
    reduced_array=np.asarray(reduced_array)
    return reduced_array
def h_stack(X,Y):
    return np.hstack((X,Y))
def extend_array(X,Y,order):
    ones=[]
    for j in range(0,len(X)):
        to_append=X[j]
        for i in range(2,order):
            to_append=h_stack(to_append,np.power(Y[j],i))
        ones.append(to_append)
    ones=np.asarray(ones)
    return ones
def make_array(X):
    ones=[]
    for i in range(0,len(X)):
        ones.append([1])
    ones=np.asarray(ones)
    return ones
#Making the data pretty, we take the mean across each column(that is for each variable)
#and divide by the standard daviation for the respective column
def normalize_array(X,Y):
    average_array=np.mean(Y,axis=0)
    std=np.std(Y,axis=0)
    for i in range(0,len(X)):
        for j in range(1,len(X[0])):
            X[i][j]=(X[i][j]-average_array[j])/std[j]
    X=np.asarray(X)
    return X
def modify_array(X,Y):
    reduced_data=reduce_array(X)
    reduced_data_2=reduce_array(Y)
    sin_data=np.sin(reduced_data)
    sin_data_2=np.sin(reduced_data_2)
    made_data=make_array(X)
    extended_data=extend_array(made_data,reduced_data,3)
    normalized_data=normalize_array(extended_data,extended_data)
    modified_data=h_stack(reduced_data,sin_data)
    return modified_data

train=modify_array(X_train,X_train)
test=modify_array(X_test,X_test)
# Fit model and predict test values 
XX = np.dot(train.T,train)
Xt = np.dot(train.T,y_train)
w = np.linalg.solve(XX,Xt)
y_pred=np.dot(test,w)
for i in range(0,len(y_pred)):
    y_pred[i]=2**y_pred[i]
    # Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")
