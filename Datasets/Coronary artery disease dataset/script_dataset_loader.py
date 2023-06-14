import numpy as np
import scipy as sp
import pandas as pd
import os

from sklearn.preprocessing import minmax_scale
from sklearn.impute import KNNImputer
from scipy import stats
from collections import Counter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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



with open(r'hungarian.data') as f:
    Hungrarian = f.read().splitlines()

with open(r'switzerland.data') as f:
    Switzerland = f.read().splitlines()

with open(r'long-beach-va.data') as f:
    Long_beach_va = f.read().splitlines()

with open(r'cleveland_cut.data') as f:
    cleveland = f.read().splitlines()
    #try:
    #    cleveland = f.read().splitlines()
    #except UnicodeDecodeError:
    #    print(f.read())


data_h = crea_dataset( Hungrarian, 'name')
data_s = crea_dataset( Switzerland, 'name')
data_l = crea_dataset( Long_beach_va, 'name')
data_c = crea_dataset( cleveland, 'name')

data = data_c + data_h + data_s + data_l 
print(len(data_c + data_h))

print(checker(data_h), checker(data_s), checker(data_l), checker(data_c), checker(data))
print(len(data_h), len(data_s), len(data_l), len(data_c))

#3  (age)  #4 (sex)  #9  (cp)  #10 (trestbps)  #12 (chol)  #16 (fbs)  #19 (restecg)   
#32 (thalach)  #38 (exang) #40 (oldpeak) #41 (slope) #44 (ca) #51 (thal) 
informative_attributes = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40] # in python list start in 0 not 1


if checker(data) == "correct":
    df_ = pd.DataFrame.from_records(data)
    df_.to_csv(r'data.csv', index=False)
    df_h = pd.DataFrame.from_records(data_h)
    df_h.to_csv(r'data_h.csv', index=False)
    df_s = pd.DataFrame.from_records(data_s)
    df_s.to_csv(r'data_s.csv', index=False)
    df_l = pd.DataFrame.from_records(data_l)
    df_l.to_csv(r'data_l.csv', index=False)
    df_c = pd.DataFrame.from_records(data_c)
    df_c.to_csv(r'data_c.csv', index=False)



X = data
df_for_analisis = pd.DataFrame.from_records(X)

dataset_name = ""

list_of_index = np.arange(len(X[0])).tolist() # Empieza en 0   # Array must be a list

Y = [ele.pop(57) for ele in X]
#print(Counter(Y))
Y = [ele if ele != 4 else 3 for ele in Y ]
#print(Counter(Y))
list_of_index.remove(57)

#df_ = pd.DataFrame.from_records(X) 

# basic statistics
nans = [np.sum((df_for_analisis[i] == -9.0))/len(df_for_analisis) for i in range(len(df_for_analisis.mean().tolist()))]
print("Percentage of Nan by columns:")
print(np.around(nans,2).tolist())

deleted_indexes = [i for i in range(len(nans)) if nans[i] > 0.25]
print("deleted indexes: ", deleted_indexes)
deleted_indexes.append(19)
deleted_indexes.append(20)
deleted_indexes.append(21)
deleted_indexes.append(54)
deleted_indexes.append(55)
deleted_indexes.append(56)
deleted_indexes.append(1)
deleted_indexes.append(0)
deleted_indexes.sort()
print("deleted indexes: ", deleted_indexes)

X = np.delete(X, deleted_indexes, axis=1) # delete rows with more than 0.25 of Nan
X = np.delete(X, -1, axis=1)              # delete name column


[list_of_index.remove(ele) for ele in deleted_indexes]
list_of_index.remove(75)


print("list of index: ")
print(list_of_index)
print("Informative index: ")
print(informative_attributes)


X = np.array(X, dtype='float').tolist()

imputer = KNNImputer(missing_values = -9.0, n_neighbors=5)
X = imputer.fit_transform(X)

print(np.mean(X, axis=0))

# basic statistics
print("Mean by columns (In order. After imputing):")
print(np.around(df_for_analisis.mean(),2).tolist()) #string features are avoided
print("Variance by columns (In order. After imputing):")
print(np.around(df_for_analisis.var()).tolist())
varianza = df_for_analisis.var().tolist()

min_max_index = []
binary_index = []
z_score_index = []
i = 0
print(len(list_of_index))
for indx in list_of_index:

    z = Counter(X[:,i])
    print
    
    if len(list(z.keys())) == 2:
        binary_index.append(indx)
        i = i+1
    else:
        X[:,i] = stats.zscore(X[:,i])
        z_score_index.append(indx)
        i = i + 1

        #if len(list(z.keys())) > 10:
            #X[:,i] = stats.zscore(X[:,i])
            #z_score_index.append(indx)
            #i = i + 1

        #else:
            #X[:,i] = minmax_scale(X[:,i])
            #min_max_index.append(indx)
            #i = i + 1


print("Binary index: ")
print(binary_index)
print("Min Max index: ")
print(min_max_index)
print("Z score index: ")
print(z_score_index)

#X = stats.zscore(X, axis=1)
#X = minmax_scale(X)

print(X[0])
np.save(r'attributes2' + dataset_name + '.npy', np.array(X))
#np.save(r'target_join' + dataset_name + '.npy', np.array(Y))
np.save(r'index2' + dataset_name + '.npy', np.array(list_of_index))
np.save(r'inf_index2.npy', np.array(informative_attributes))




