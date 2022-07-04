#!/usr/bin/env python
# import the required packages here
'''
Voted Perceptron 
    - Applying the Voted Perceptron algorithm on the classification task
    
   Initiate k=1, c_1 = 0, w_1 = 0, t = 0;
   while t <= T do
       for each training example (x_i, t_i) do
           if t_i (w_k x_i) <= 0 then
                w_k+1 = w_k + t_i x_i; c_k+1 = 1;
                k=k+1
           else
                c_k += 1;
           end 
       end
       t = t + 1; 
   end

'''
import numpy as np
from sklearn.model_selection import train_test_split

# Importing difflib
import difflib
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

    Xtrain_data  = np.loadtxt(Xtrain_file,delimiter=",", dtype = 'int')
    Ytrain_data  = np.loadtxt(Ytrain_file,  dtype = 'int')
    test_data  = np.loadtxt(test_data_file, delimiter=",", dtype = 'int')
   
    
    Ytrain_data = np.array(Ytrain_data, dtype = int)
    Xtrain_data = np.array(Xtrain_data, dtype = int )
    Xtest_data = np.array(test_data , dtype = int)

    Xtrain_data, Xtest_data , Ytrain_data, Ytest_data  = train_test_split( 
        Xtrain_data ,  Ytrain_data , test_size = 0.1, random_state = 42)
    # your algorithm
    # 1 percent
    Xtrain_data, X , Ytrain_data, Y  = train_test_split( 
        Xtrain_data ,  Ytrain_data , test_size = 0.99, random_state = 42)
    T = len(Xtrain_data)
    K= 0
    c = np.empty(T, dtype= np.int16) 
    w = np.empty((T ,  len(Xtrain_data[0]))) 
    c[0] = 0
    w[0] = np.full((1, len(Xtrain_data[0])), 0) 
   
   
    # print(len(Ytrain_data))
    for i in range (T):
        total = 0 
        if (  Ytrain_data[i] == 0):
            Ytrain_data[i] = -1 
        if ( Ytrain_data[i] * np.dot(w[K],Xtrain_data[i])  <= 0):
            total = 0 
            # for j in range (len(Xtrain_data[i])):
            #     total +=  (Xtrain_data[i][j] * Ytrain_data[i] ) 
            
            w[K+1] = w[K] + Xtrain_data[i] * Ytrain_data[i] 
            # np.full((1, len(Xtrain_data[0])), w[K] + total ) 
           
            c[K+1] = 1 
            K = K + 1 
        else: 
            c[K] += 1
    # c[0] = 0 
    # w[0] = 0 
    # sign function, the function that returns -1, 1 if the input is < or > than 0
    # sign(wkx)
    prediction = np.empty( len(Xtest_data), dtype= np.int16) 
    
    for i in range(len(Xtest_data)):
        total = 0 
        for k in range(K):
            sign = 0
            # for j in range (len(Xtrain_data[i])):
            #     sign += w[k] * Xtrain_data[i][j]
            
            if (np.dot(w[k] ,Xtest_data[i]) < 0):
               
                sign = -1 
                total -= c[k]
                
            else:
               
                sign = 1
                total += c[k]
               
        
        if (total < 0):
        
            prediction[i] = 0
        else:
            prediction[i] = 1

    # save your predictions into the file pred_file
    # test_data = np.loadtxt(test_input_dir,skiprows=0)
    # print(test_data.shape)
    # [num, _] = test_data.shape
   
    # np.zeros((num, 1), dtype=np.int16)
    # write the prediction into a single file named predfile
    np.savetxt(pred_file, prediction, fmt ='%ld', delimiter= ",") 
   
    # Find and print the diff:
    errors = 0 
    lines = 0 
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

