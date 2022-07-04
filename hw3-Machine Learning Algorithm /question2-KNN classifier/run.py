#!/usr/bin/env python
# import the required packages here
import numpy as np
from sklearn.model_selection import train_test_split


'''
KNN classifier
    - 
'''
def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
    '''The function to run your ML algorithm on given datasets, generate the predictions and save them into the provided file path
    Parameters
    ----------
    Xtrain_file: string
    the path to Xtrain csv file
    Ytrain_file: string
    the path to Ytrain csv file
    test_data_file: string
    the path to test data csv file
    pred_file: string
    the prediction file to be saved by your code. You have to save your predictions into this file path following the same format of
    Ytrain_file
    '''
    ## your implementation here
    # read data from Xtrain_file, Ytrain_file and test_data_file

    Xtrain_data  = np.loadtxt(Xtrain_file,delimiter = ",", dtype = 'float')
    Ytrain_data  = np.loadtxt(Ytrain_file,  dtype = 'float')
    test_data  = np.loadtxt(test_data_file, delimiter=",", dtype = 'float')
   
    
    Ytrain_data = np.array(Ytrain_data, dtype = float)
    Xtrain_data = np.array(Xtrain_data, dtype = float )
    Xtest_data = np.array(test_data , dtype = float)

    Xtrain_data, Xtest_data , Ytrain_data, Ytest_data  = train_test_split( 
        Xtrain_data ,  Ytrain_data , test_size = 0.1, random_state = 42)
    # your algorithm
    # 1 percent

    T = len(Xtrain_data)
    k = 8
    prediction = list()
    
    
    for i in range(len(Xtest_data)):
        dist = list()
        for x in range(len(Xtrain_data)):
            distance = 0
            for j in range(len(Xtest_data[i])):
                distance += (Xtest_data[i][j] -  Xtrain_data[x][j])**2
            euclidean = distance**(1/2)
            dist.append((Ytrain_data[x],  euclidean))
        dist = sorted(sorted(dist, key = lambda x : x[0]), key = lambda x : x[1])  
        neighbors = list()
        # print(dist)
        for i in range(k):
            # print("neighbor",dist[i][0])
            neighbors.append( dist[i][0])
        prediction.append(max(set(neighbors), key=neighbors.count))
        
    # write the prediction into a single file named predfile
    # print(prediction)
    np.savetxt(pred_file, prediction, fmt ='%.1f', delimiter= ",") 
   
    # Find and print the diff:
    errors = 0 
    lines = 0 
    print(Ytest_data)
    for i in range (len(prediction)):
        lines += 1 
        if(prediction[i]!= Ytest_data[i] ):
        
            errors += 1 
    
    print("accuracy is ", 1 - (errors/lines))
    # define other functions here
if __name__ == "__main__":
    # the path to the training dataset txt file
    Xtrain_file = 'Xtrain.csv'
    Ytrain_file= 'Ytrain.csv'
    # X  = np.loadtxt(Xtrain_file,skiprows=0)
    # Y  = np.loadtxt(Ytrain_file,skiprows=0)
    
    # print(X.shape)
    # print(Y.shape)
   
    # f = open('test.csv')
    # writer = csv.writer(f)
    # writer.writerow(X_test)
    test_data_file = 'Xtrain.csv'
    #the file name of your output prediction file
    pred_file = 'result'
    run(Xtrain_file, Ytrain_file, test_data_file, pred_file)

