

## Auxiliar document

import numpy as np



def to_list(Z):

    if isinstance(Z, list):
        None


    elif isinstance(Z, np.ndarray):  
        Z.tolist()
        
    else:
        print("Error, insert a valid list or array ")
        exit()

    return Z


def binary_change(Y_tr_, Y_cal_):

    for i in range(len(Y_tr_)):
        if Y_tr_[i] == 0:
            Y_tr_[i] = -1

    for i in range(len(Y_cal_)):
        if Y_cal_ [i] == 0:
            Y_cal_ [i] = -1

    y_classes_name, Y_tr_ = np.unique(Y_tr_, return_inverse=True)
    
    print(y_classes_name )

    return Y_tr_, Y_cal_, y_classes_name

