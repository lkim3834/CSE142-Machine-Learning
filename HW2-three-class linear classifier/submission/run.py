import numpy as np
import sys
'''
train_input_dir : the directory of training dataset txt file. For example 'training1.txt'.
train_label_dir :  the directory of training dataset label txt file. For example 'training1_label.txt'
test_input_dir : the directory of testing dataset label txt file. For example 'testing1.txt'
pred_file : output directory 
'''

def run (train_input_dir,train_label_dir,test_input_dir,pred_file):
    # Reading data
    #load the training data examples located at train_dir
    train_data  = np.loadtxt(train_input_dir,skiprows=0)
    #Train a three-class linear classifier on the training data. 
    # print(train_data)
    # Lines = train_data.readlines()
    count = 0 
    length = 0
    centroid = [0 for i in range(5)]
    for line in train_data: 
        count += 1
        length = len(line)
        for i in range(  length):
            centroid[i] += line[i]
       
        # print("Line{}: ".format(count))
    for i in range(  length):
        centroid[i] = (centroid[i] )/count 
        # print( i, centroid[i])
    #3
    func = [0 for i in range(  length)]
    for i in range(  length):
        if (i + 1 ==   length):
            func[i] = (centroid[0] + centroid[i]) /2 
        else: 
            func[i] = (centroid[i] + centroid[i+1]) /2 
    
    # if centroid[i] is less than the func[i] it means dot less than or equal to fun[i] classified as i classs
    # else : dot bigger or eqaul to func[i] are classifed as class i 
    labels = [0 for i in range(  length)]
    # pair of (func[0] , 0) and (0, func[1])
    weights_1 = [0 for i in range( 2)]
    # pair of (func[1] , 0) and (0, func[2])
    weights_2 = [0 for i in range( 2)]
    # pair of (func[0] , 0) and (0, func[2])
    weights_3 = [0 for i in range( 2)]
    
    weights_1[0] = func[0]/func[1]
    weights_1[1] = (pow(func[1],2) - pow(func[0],2) ) / (2*(func[1]))
    weights_2[0] = func[1]/func[2]
    weights_2[1] = (pow(func[2],2) - pow(func[1],2) ) / (2*(func[2]))
    weights_3[0] = func[0]/func[2]
    weights_3[1] = (pow(func[2],2) - pow(func[0],2) ) / (2*(func[2]))
    print(weights_1[0], weights_1[1])
    c =  [0 for i in range(  length)]
    
    result = [0 for i in range(  count)]
    cou = 0 
    for line in train_data: 
            freq = [0 for i in range(  length)]
            if(line[1]-(line[0] *  weights_1[0] +  weights_1[1]) >= 0 ):
                c[0] = 0 
            else:
                c[0] = 1
            if(line[2]- (line[1] *  weights_2[0] +  weights_2[1]) >= 0 ):
                c[1] = 1 
            else:
                c[1] = 2
            if(line[0]-(line[2] *  weights_3[0] +  weights_3[1] )>= 0 ):
                c[2] = 2
            else:
                c[2] = 0
            for i in range(  length):
                freq[c[i]] += 1 
            
            max_value = (np.max(freq))
            # print(max_value)
            # print(freq)
            for i in range(  length):
                if(max_value == freq[i]):
                    result[cou] = i
                    # print(i)
                    break
            cou += 1
           
             
            


    test_data = np.loadtxt(test_input_dir,skiprows=0)
    print(test_data.shape)
    [num, _] = test_data.shape
    
    prediction = np.reshape(result, (num,1))
    # np.zeros((num, 1), dtype=np.int16)
    # write the prediction into a single file named predfile
    np.savetxt(pred_file, prediction, fmt ='%ld', delimiter= ",") 
    
  

    
if __name__ == "__main__":
    # the path to the training dataset txt file
    train_input_dir = 'training1.txt'

    train_label_dir = 'training1_label.txt'
    test_input_dir = 'testing1.txt'
    #the file name of your output prediction file
    pred_file = 'result'
    run(train_input_dir,train_label_dir,test_input_dir,pred_file)
