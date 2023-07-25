

## Authors: Marcos López De Castro 
#           Alberto García Galindo
#           Rubén Armañanzas

"""Conformal recursive feature elimination for feature selection"""



import numpy as np
from numbers import Integral

from sklearn.svm import SVR, LinearSVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils._param_validation import HasMethods, Interval, Real
from sklearn.base import BaseEstimator, clone
from sklearn.multiclass import OneVsRestClassifier

from CRFE._crfe_utils import to_list, binary_change




class CRFE(BaseEstimator):


    """

    Parameters
    -------------

    estimator : Supervised learning estimator. 
        Must have a fit method. 
        Must provide "coef_" info.

        
    features_to_select : The number of features to select.
        Integer. Default = 1. 
        Is the absolute number of features to select. 
                         
        
    Lambda :  Multi-class parameter. 
        Default = 0.5. Float between 0 and 1. 
        Controls the weight of the "one" class in "OneVSRest" strategy.


    stopping_activated : If the beta-based stopping criteria is considered.


    epsilon : Beta-based stoping criteria. 
        Float. Ranges from 0 to 1. 



    Attributes
    ------------

    
    idx_features_ : List with the index of the selected features.
    

    idx_betas_ : List with betas linked to each feature when the recursive process is over.


    estimator_ : The fitted estimator used to select features.


    classes_ : Classes labels available when estimator is a classifier.

    """

    
    _parameter_constraints: dict = {
        "estimator" : [HasMethods(["fit"])],
        "features_to_select": [Interval(Integral, 0, None, closed="neither")],
        "Lambda":[Interval(Real, 0, 1, closed="both")],
        "epsilon":[Interval(Real, 0, 1, closed="both")]
    }


    def __init__(
            self,
            estimator = None,                       # Estimator 
            features_to_select = 1,                 # Number of features to select  
            Lambda = 0.5,                           # Parameter
            stopping_activated = False,             # If Beta-based stopping criteria
            epsilon = 0.4
    ):
        
        self.estimator = estimator
        self.features_to_select = features_to_select
        self.features = []
        self.betas = []
        self.Lambda = Lambda
        self.stopping_activated = stopping_activated
        self.epsilon = epsilon 


    def beta_based_stopping_criteria(self,beta):

        if self.stopping_activated == False:
            pass

        else:

            beta_sum =  np.sum(beta)/len(beta)   # mean of betas

            if len(self.stopping_list) > self.stopping_boost:
                del self.stopping_list[0]                     # delete the older computed element of the list if is larger than the pre-established set size

            self.stopping_list.append(beta_sum)

            if starting_flag == True:
                stopping_list_ = [ beta_sum for _ in range(2)]      # initialize the list
                x_der_ = [ i for i in range(len(stopping_list_))]   # list of "points" to compute the numeric derivative
                   
                self.stopping_list_grad_2.append(np.gradient(np.gradient(stopping_list_, x_der_), x_der_).tolist()[-1])  # Second derivative

                starting_flag = False

            else:
                x_der = [ i for i in range(len(self.stopping_list))]      # list of "points" to compute the numeric derivative
                    
                second_deriv = np.gradient(np.gradient(self.stopping_list, x_der), x_der).tolist()

                #not in the subsequent loop becasuse we need to append them to the list
                self.stopping_list_grad_2.append(second_deriv[-1]) 

                #correction of the derivatives

                for i in range(-2, round((-len(self.stopping_list)-1)/2), -1):
                        
                    self.stopping_list_grad_2[i] = second_deriv[i]

            n_sigmas = 5
            self.std_dev.append(np.std(self.stopping_list_grad_2[-len(self.stopping_list):]))    # compute the std deviation only of the "self.stopping_boost" latest elements

            if len(self.stopping_list_grad_2) >  round(self.epsilon/2):
                    
                if self.stopping_list_grad_2[-2] < -n_sigmas*(self.std_dev[-2]):  

                    return True

        return False
    

    def recursive_elimination( self, X_tr, Y_tr, X_cal, Y_cal):

        list_of_index = np.arange(len(X_tr[0])).tolist()          # Empieza en 0   # Array must be a list
        n = len(list_of_index)                                    # Number of features
        self.Lambda_p = (1-self.Lambda) / (len(self.classes_)-1)  # Theoretical parameter

        self.stopping_boost = int(n*self.epsilon)                 # List needed for the beta-based-stopping criteria
        self.stopping_list = []
        self.starting_flag = True
        self.stopping_list_grad_2 = []
        self.std_dev = []


        X_tr = np.insert(X_tr , 0, list_of_index, 0)  # Insert a header in the array
        X_cal = np.insert(X_cal , 0, list_of_index, 0)
        

        ### recursive elimination loop  ###

        list_of_deleted_indexes = []
        while n != self.features_to_select:

            head = X_tr[0]
            X_tr = np.delete(X_tr , 0, 0) # delete the head
            X_cal = np.delete(X_cal , 0, 0)


            ## fit the classifier
            
            if len(self.classes_) == 2:

                self.estimator.fit(X_tr, Y_tr)
                w = self.estimator.coef_
                bias = self.estimator.intercept_

            else: 
                if isinstance(self.estimator, OneVsRestClassifier) == True:
                    self.estimator.fit(X_tr, Y_tr)

                else:
                    self.estimator = OneVsRestClassifier(self.estimator)
                    self.estimator.fit(X_tr, Y_tr)

                w = [self.estimator.estimators_[i].coef_.tolist()[0] for i in range(len(self.estimator.estimators_))]
                bias = [self.estimator.estimators_[i].intercept_.tolist()[0] for i in range(len(self.estimator.estimators_))]

            ## compute betas 
            
            if len(self.classes_) == 2:
                
                beta = []
                for j in range(n):

                    beta_i = 0
                    for i in range(len(X_cal)):
                        
                        #print(Y_cal[i])
                        beta_ij = w[0][j] * Y_cal[i] * X_cal[i][j]
                        beta_i = beta_i - beta_ij

                        i = i + 1

                    beta.append(beta_i)

            else:

                beta = []
                for j in range(n):
                    
                    beta_i = 0
                    for i in range(len(X_cal)):

                        w_aux = np.delete(w, Y_cal[i], axis = 0 )
                        bias_aux = np.delete(bias, Y_cal[i] ) 
                        
                        beta_ij = self.Lambda * 1. * w[Y_cal[i]][j]*X_cal[i][j] + self.Lambda_p * (-1.) * np.sum( w_aux[:,j] )*X_cal[i][j]
                        beta_i = beta_i - beta_ij
                    
                        i = i + 1

                    beta.append(beta_i) 

                
            deleted_index = beta.index(max(beta)) # Indice en w del minimo. la posición del indice coincide con la del head
            list_of_deleted_indexes.append(deleted_index)
            
            stopping_flag = self.beta_based_stopping_criteria(beta)
            if stopping_flag == True:
                break
            
            X_tr = np.insert(X_tr , 0, head , 0)
            X_cal = np.insert(X_cal , 0, head , 0)

            X_tr = np.delete(np.array(X_tr), deleted_index , axis = 1 )
            X_cal = np.delete(np.array(X_cal), deleted_index , axis = 1 )
            
            list_of_index = np.arange(len(X_tr[0])).tolist() # Empieza en 0   # Array must be a list

            n = len(list_of_index)
            
            print("Remaining indices: ", n )

            self.features =  X_tr[0].astype(int)
            beta.pop(deleted_index )
            self.betas = beta
            
        #if self.stopping_activated == False:
            #self.features = X_tr[0].astype(int)   #las que se quedan
        

        return None


    def fit(self, X_tr, Y_tr, X_cal, Y_cal):
       
        #self.validate_params(self._constrains)   # here we check if params are correct
        self._validate_params()

        tags = self._get_tags()
        
        X_tr, Y_tr = self._validate_data(X_tr, Y_tr,
                                        accept_sparse="csc",
                                        ensure_min_features=2,
                                        force_all_finite= not tags.get("allow_nan", True),
                                        multi_output=True)
        
        X_cal, Y_cal = self._validate_data(X_cal, Y_cal,
                                        accept_sparse="csc",
                                        ensure_min_features=2,
                                        force_all_finite=not tags.get("allow_nan", True),
                                        multi_output=True)


        X_tr = to_list(X_tr)
        Y_tr = to_list(Y_tr)
        X_cal = to_list(X_cal)
        Y_cal = to_list(Y_cal)


        if len(X_tr[0]) != len(X_cal[0]):
            print("ValueError: X training and X features number not the same")
            exit()


        if list(np.unique(Y_tr, return_inverse=True)[0]).sort() != list(np.unique(Y_cal, return_inverse=True)[0]).sort():
            print("ValueError: All classes in training are not present in calibration")
            exit()

        self.classes_, Y_tr = np.unique(Y_tr, return_inverse=True)

        if len(self.classes_) == 2:
            Y_tr, Y_cal, self.classes_ = binary_change(Y_tr, Y_cal)      

        
        self.recursive_elimination( X_tr, Y_tr, X_cal, Y_cal)  #  CRFE 

        #check_is_fitted(self.estimator)   ## Check if fit was called.

        # Set final attributes  
        self.idx_features_ = self.features
        self.idx_betas_ = self.betas
        self.estimator_ = clone(self.estimator)
       
        return self 
    
