import numpy as np
import scipy as sp
import pandas as pd
import os

from sklearn.preprocessing import minmax_scale
from sklearn.impute import KNNImputer

def crea_dataset(lista,stop):

    dataset = []
    for sublista in lista:

        pre_number = ""
        b = []
        i = 0
        lon = len(sublista)
        for ele in sublista:

            if ele != " ":
                
                pre_number = pre_number + ele
               
            else: 
                
                b.append(float(pre_number))
                pre_number = ""

            
            if pre_number == 'name':
                
                b.append(pre_number)
                pre_number = ""
            
            elif i == lon-1:
                
                b.append(float(pre_number))
                pre_number = ""

            else:
                pass

           
            i = i + 1

        
                    
        dataset.append(b)

    data_fattened = [subele for ele in dataset for subele in ele ]

    dataset = []
    aux = []
    for ele in data_fattened:
        
        if ele != stop:
            aux.append(ele)
        else:
            aux.append(ele)
            dataset.append(aux)
            
            aux = []

    return dataset

def checker(data):

    size_old = len(data[0])
    for i in range(1,len(data)):

        size_new = len(data[i])

        if size_old == size_new:
            status = "correct"
            size_old = size_new
        else:
            print(i)
            status ="wrong"

    return status



with open(r'C:\Users\Marcos\Desktop\Year1\Heart Disease Data Set\hungarian.data') as f:
    Hungrarian = f.read().splitlines()

with open(r'C:\Users\Marcos\Desktop\Year1\Heart Disease Data Set\switzerland.data') as f:
    Switzerland = f.read().splitlines()

with open(r'C:\Users\Marcos\Desktop\Year1\Heart Disease Data Set\long-beach-va.data') as f:
    Long_beach_va = f.read().splitlines()

with open(r'C:\Users\Marcos\Desktop\Year1\Heart Disease Data Set\cleveland_cut.data') as f:
    cleveland = f.read().splitlines()
    #try:
    #    cleveland = f.read().splitlines()
    #except UnicodeDecodeError:
    #    print(f.read())


data_h = crea_dataset( Hungrarian, 'name')
data_s = crea_dataset( Switzerland, 'name')
data_l = crea_dataset( Long_beach_va, 'name')
data_c = crea_dataset( cleveland, 'name')

data = data_h + data_s + data_l + data_c

print(checker(data_h), checker(data_s), checker(data_l), checker(data_c), checker(data))


#3  (age)  #4 (sex)  #9  (cp)  #10 (trestbps)  #12 (chol)  #16 (fbs)  #19 (restecg)   
#32 (thalach)  #38 (exang) #40 (oldpeak) #41 (slope) #44 (ca) #51 (thal) 
informative_attributes = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50] # in python list start in 0 not 1


if checker(data) == "correct":
    df = pd.DataFrame.from_records(data)
    df.to_csv(r'C:\Users\Marcos\Desktop\Year1\Heart Disease Data Set\data.csv', index=False)

X = data
list_of_index = np.arange(len(X[0])).tolist() # Empieza en 0   # Array must be a list
Y = [ele.pop(57) for ele in X]
list_of_index.remove(57)

#df_ = pd.DataFrame.from_records(X) 

# basic statistics
print("Mean by columns:")
print(df.mean().tolist()) #string features are avoided
print("Variance by columns:")
print(df.var().tolist())
nans = [np.sum((df[i] == -9.0))/len(df) for i in range(len(df.mean().tolist()))]
print("Percentage of Nan by columns:")
print(nans)

deleted_indexes = [i for i in range(len(nans)) if nans[i] > 0.5]


X = np.delete(X, deleted_indexes, axis=1) # delete rows with more than 0.5 of Nan
X = np.delete(X, -1, axis=1)              # delete name column

[list_of_index.remove(ele) for ele in deleted_indexes]
list_of_index.remove(75)

print("list of index: ")
print(list_of_index)

X = np.array(X, dtype='float').tolist()


from sklearn.impute import KNNImputer
imputer = KNNImputer(missing_values = -9.0, n_neighbors=2)
X = imputer.fit_transform(X)
from sklearn.preprocessing import minmax_scale
X = minmax_scale(X)


np.save(r'C:\Users\Marcos\Desktop\Year1\Heart Disease Data Set\attributes.npy', np.array(X))
np.save(r'C:\Users\Marcos\Desktop\Year1\Heart Disease Data Set\target.npy', np.array(Y))
np.save(r'C:\Users\Marcos\Desktop\Year1\Heart Disease Data Set\features.npy', np.array(list_of_index))