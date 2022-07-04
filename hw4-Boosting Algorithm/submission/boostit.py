import numpy as np

class BoostingClassifier:
    """ Boosting for binary classification.
    Please build an boosting model by yourself.

    Examples:
    The following example shows how your boosting classifier will be used for evaluation.
    >>> X_train, y_train = load_train_dataset() # we ignore the process of loading datset
    >>> X_test, y_test = load_test_dataset()
    >>> clf = BoostingClassifier().fit(X_train, y_train)
    >>> y_pred =  clf.predict(X_test) # this is how you get your predictions
    >>> evaluation_score(y_pred, y_test) # this is how we get your final score for the problem.

    """
    def __init__(self):
        # initialize the parameters here
        model = list()
    

    def fit(self, X, y):
        """ Fit the boosting model.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            The input samples with dtype=np.float32.
        
        y : { numpy.ndarray } of shape (n_samples,)
            Target values. By default, the labels will be in {-1, +1}.

        Returns
        -------
        self : object
        """
        
        centroid = [0.0 for i in range(5)]
        dim = 0 
        # ensemble size T can be changed by me 
        T = 4
        # k = D = num of training points 
        k = len(X)
        
        for line in X:
            dim = len(line)
            break
        # weights matrix for Tk 
        weights = [[0.0 for x in range(T+2)] for y in range(k)]
        # plus 1 for t value 
        M = [[0.0 for x in range(T+1)] for y in range(dim + 1)]
        for i in range(k):
            weights[i][1] = 1/k
            
        
        length = 0
        # class1 is for the class classified as 1 
        class1_examplar = [0.0 for i in range(dim)]
        # class2 is for the class classified as -1
        class2_examplar = [0.0 for i in range(dim)]
        # this is to produce a model M_t 
        # this is to get class_examplar 
        a = [0 for i in range(T+1)]
        # print(y)
        
          
        for t in range(1, T+1):
            print("Iteration ",t , ":")
            for j in range(k):
                if(y[j] == 1):
                    for i in range(dim):
                        class1_examplar[i] += (weights[i][t] * X[j][i])
                    
                
                else: 
                    for i in range(dim):
                        class2_examplar[i] += (weights[i][t] * X[j][i])

            # midpoint that the function will pass through
            # mid_point = [0 for i in range(dim)]
           
            # for i in range(dim):
            #     mid_point[i] = (class1_examplar[i] + class2_examplar[i])/2
         
            # print("this is mid point for ", t , ":", mid_point)
            # model for M[dim][t]
            # perpendicular to the line that passess class1_examplar & class2_examplar , pases midpoint
            # for i in range(dim):
            #     M[i][t] = 
            for i in range(dim):
                M[i][t] = (class1_examplar[i] - class2_examplar[i])
               
            t_for_model = 0 
            # print("class1:", class1_examplar)
            # print("class2: ", class2_examplar)
            for i in range(dim):
                t_for_model += (class1_examplar[i] ** 2) - (class2_examplar[i] ** 2)
            M[dim][t]  = (t_for_model) / 2
            # print("t is ", M[dim][t])
            # model = w[0]x_0 + w[1]x_1 + ... + w[d]x_d + t            
             
                

            e =  0 

            #######calculate the weighted error################
            # if func[1] * X[j][1] + func[2] * X[j][2] + ... +  func[dim-1] * X[j][dim-1] > 0, if y[j] != 1, e += 1
            # else, if y[j] != -1 , e += 1 
            #misclassified instances
            m_class = []
            #classified instances
            r_class = []
            for j in range(k):
                classify = 0 
                for i in range(dim ):
                    classify += (M[i][t] * X[j][i]) 
                classify -= M[dim][t]
            
                
                if (classify > 0 ):
                    if (y[j] != 1 ):
                        # misclassified
                        e += 1 
                        m_class.append(j)
                    else: 
                        r_class.append(j)
                        
                else: 
                    if (y[j] != -1 ):
                        # misclassified
                        e +=1 
                        m_class.append(j)
                    else: 
                        r_class.append(j)
            ###################################################
            # print("this is e", e)
            e = e/k
            # print(m_class)
            # print(r_class)
            print("Error = ", e )
            # print("t is",t)
            if (e >= (1/2)):
                T = t - 1 
                
                break 
            a[t] = (1/2)* np.log( (1-e)/ e)
            print("Alpha = ", a[t])
            # for i in range(dim):
            #     class1_examplar[i] += weights[i][t] * X[j][i]
            # misclassified instances 
            c = 0 
            factor  = 0 
            for i in range (len(m_class)) :
                c+= 1
                weights[m_class[i]][t+1] = (weights[m_class[i]][t])/(2*e)
            factor = 1/(2*e)
            print("Factor to increase weights = ", factor)
          
            # correctly classified instances 
            for i in range (len(r_class)) :
                c += 1
                weights[r_class[i]][t+1] = (weights[r_class[i]][t])/(2*(1-e))
            factor = 1/(2*(1-e))
            print("Factor to decrease weights = ", factor  )
        sum = [0.0 for i in range(dim+1)]
        # print(T)
        # print(t)
        c = 0 
        for t in range(1, T+1 ):
            c+= 1
            for i in range(dim+1 ):
                sum[i] += a[t] * M[i][t]  
                # print(sum[i])
        # print(c)
        self.model = sum 
        # pass
       
        return self

    def predict(self, X):
        """ Predict binary class for X.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)

        Returns
        -------
        y_pred : { numpy.ndarray } of shape (n_samples)
                 In this sample submission file, we generate all ones predictions.
        """
        
        result = np.ones(X.shape[0], dtype=int)
        model = self.model 
        k = len(X)
        dim = 0 
        for line in X:
            dim = len(line)
            break
        for j in range(k):
                classify = 0 
                for i in range(dim ):
                    classify += (model[i] * X[j][i]) 
                classify -= model[dim]
                if (classify > 0 ):
                    result[j] = 1 
                else: 
                    result[j] = -1

        return result

