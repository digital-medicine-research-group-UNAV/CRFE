
from CRFE._crfe import CRFE
from sklearn.svm import SVR, LinearSVC
import numpy as np
from scipy.stats import zscore
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pandas as pd
import numpy as np
import itertools
#import Settings

from scipy.stats import zscore
from collections import Counter


from scipy import stats


def z_scores(df, name, parameter = True):

    if parameter == True:
        df_name = df[name] 
        zscores = df.select_dtypes(include = "number").apply(zscore, axis=1)
        df_zscores = pd.concat([df_name, zscores], axis=1)

        # Delete Nans 
        Nan = list(df_zscores['RNA-9b6b1b307aa2'].isnull())
        #print(Nan)
        indx = [i for i in range(len(Nan)) if Nan[i] == True]
        #print(len(indx))
        df_zscores = df_zscores.drop(index=indx)
        df_zscores = df_zscores.reset_index(drop=True)

        df_zscores.to_csv(r'/home/mlopezdecas/Marcos/urothelial-carcinoma/Code/DATASETS/IMvigor010/z_scored_tpm.csv', index=False, header=True)
    else:
        df_zscores = df

    return df_zscores

class load_IMVIGOR010():
       

    def loader(self, arg = 1):

        df_clinical_data = pd.read_csv(r'../urothelial-carcinoma/Code/DATASETS/IMvigor010/imvigor010_clinical_ctDNA.csv', keep_default_na=False)
        #df_transcripts = pd.read_csv(r'DATASETS/IMvigor010/EGAD00001007653/EGAF00005279592/wo29636_tpm_20210423.csv')
        df_transcripts = pd.read_csv(r'../urothelial-carcinoma/Code/DATASETS/IMvigor010/z_scored_tpm.csv')
        
        if arg == 0:

            return df_clinical_data, df_transcripts

        else:
            
            df_transcripts = z_scores(df_transcripts, "Unnamed: 0", False) # z_scores are computed by columns
            
            return df_clinical_data, df_transcripts
    
    _alias_dicc = {} 
    def column_name(self, df):
      
        for head in df.columns:
            try:
                self._alias_dicc[head] = df[head].tolist()
            except:
                head = df.columns[0] # generalizable

        keys = list(self._alias_dicc.keys())

        return  keys



    _row_key_dicc = {}
    def alias_patient(self, keys, df): # incluir con la anterior y hacer solo una llamada
                                       # this function returns the dictionary with the alias-row relationship
        for key in keys:
    
            key_ = key.replace("-", "." ) # generalizable
           
            self._row_key_dicc[key] =  df[df['alias'] == key_].index
    
        return self._row_key_dicc

    def new_predictor(self, df_copy, df_paste, _keys, _row_key_dicc, new_attribute ):
        
        list_of_var = []
        for key in _keys:
            try:
                var = df_copy[[new_attribute[0]]].loc[[_row_key_dicc[key][0]]].iloc[0][0]
                if var == "YES":
                    list_of_var.append(0.5)
                elif var == "NO":
                    list_of_var.append(-0.5)
           
            except IndexError:             ## no consideramos el asociado a las heads
                list_of_var.append(new_attribute[0])      
          
        s = pd.DataFrame([list_of_var] ,columns = _keys)
        new_df = pd.concat([s, df_paste[:]],  ignore_index=True)
        
        return new_df


    _X = []
    _Y = []  
    def run(self, arg, classes, main_class, new_attribute, subdivs):
    

        self._df_clinical_data, self._df_transcripts = self.loader( arg )
        
        _keys = self.column_name(self._df_transcripts)      

        _row_key_dicc = self.alias_patient(_keys, self._df_clinical_data)

        self._df_transcripts = self.new_predictor(self._df_clinical_data, 
                                                  self._df_transcripts, _keys, _row_key_dicc, new_attribute )
        

        # List with subclasses
        list_of_classes = []
        for classe in classes:
            
            list_classes = [*set(self._df_clinical_data[classe].values.tolist())]
            list_of_classes.append(list_classes)
        
       
        for sub in subdivs:
            subdiv = [*set(self._df_clinical_data[sub].values.tolist())]
        
        for i in range(len(list_of_classes)):
            try:
                list_of_classes[i].remove("") # remove Nan
            except ValueError:
                pass


        ## Establecer condiciones del atributo a predecir

        _List_of_comb = []
        try:
            for ele in subdiv:
            
                aux = list_of_classes + [[ele]]
                list_of_comb = []
                for element in itertools.product(*aux):

                    list_of_comb.append(element)

                _List_of_comb.append(list_of_comb)

        except UnboundLocalError:

            for element in itertools.product(*list_of_classes):

                    _List_of_comb.append(element)

        
        _Y_index = []
        for sub_classe in sorted(list_of_classes[main_class]):
            _Y_index.append(list(self._df_clinical_data[self._df_clinical_data[classes[main_class]] == sub_classe ].index))

        
        # flate the list (why?)
        _Y_index = [ele for sub_list in _Y_index for ele in sub_list]

        
        for list_of_comb in _List_of_comb:
        
            _x, _y = [],[]
            for index_ in _Y_index:
                for key in _keys:

                    try:
                        if _row_key_dicc[key][0] == index_:

                            y_ = []
                            for classe in classes:

                                y_.append(self._df_clinical_data[classe].loc[index_]) # value of label
                                
                            k = 0
                            for comb in list_of_comb: # This for is useless
                                
                                if y_ == list(comb):

                                    y = k
                                else:
                                    k = k + 1 

                            _y.append(y_[0]) # target
                            
                            x = self._alias_dicc[key] # la key es el identificador de la muestra

                            _x.append(x) # atributes
                            
                            break
                        else:
                            pass

                    except IndexError:
                        pass

            self._Y.append(_y)
            self._X.append(_x)

         

        print("Combinaciones: \n", _List_of_comb)


        #print(self._Y[1][:20])
        return self._X, self._Y
    


IMVIGOR010 = load_IMVIGOR010()

arg = 1
df_clinical_data, df_transcripts = IMVIGOR010.loader(arg)
target, main_class, new_attribute, subdiv = [ "C1_to_C3"], 0, ["prior_neoadjuvant_chemotherapy"], ["ARM"]
X,Y = IMVIGOR010.run(arg, target, main_class, new_attribute, subdiv  )
X,Y = shuffle(X[1], Y[1], random_state= 1234)

y_classes_name = sorted(list(set(Y)))  # list with the name of the classes sortes alph
y_classes = range(len(y_classes_name))   #Generalizar

print("classes considered: ", y_classes_name )

i = 0
for ele in Y:     
    Y[i] = y_classes_name.index(ele)   # we assing to each class a number
    i = i+1


X_tr , X_cal , Y_tr, Y_cal = train_test_split( X, Y, test_size=0.5, stratify=Y)


estimator = LinearSVC(tol = 1.e-4, 
                      loss='squared_hinge',
                      multi_class = "ovr")


crfe = CRFE(estimator , features_to_select = 200)
crfe.fit(X_tr, Y_tr, X_cal , Y_cal)
print("Features selected: ", crfe.features_idx_())

