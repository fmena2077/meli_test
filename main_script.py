# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:23:53 2023

@author: franc
"""
import os
import pandas as pd
from clase_meli import modeloMeli


if __name__ == "__main__":
    print("# 1. Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    
    meli = modeloMeli()
    X_train, y_train, X_test, y_test = meli.build_dataset()
    

    #Check for balance of classes
    print( pd.Series(y_train).value_counts(normalize=True) )
    """
    It is a fairly balanced dataset (54% new, 46% used), there won't be problems of imbalance
    """


    print("# 2. Converting dictionaries to dataframe")
    lista_df = []
    for i, x in enumerate(X_train):
        print(i)
        
        dfaux = meli.dict_to_df(x, i) 
        lista_df.append(dfaux)

    df_train = pd.concat(lista_df, ignore_index=True)
    print("Train set completed!")
    

    lista_test = []
    for i, x in enumerate(X_test):
        print(i)
        
        dfaux = meli.dict_to_df(x, i) 
        lista_test.append(dfaux)
    
    df_test = pd.concat(lista_test, ignore_index=True)
    print("Test set completed!")




    print("# 3. Cleaning data and building features")
    dftrain = meli.feature_engineering(df_train, drop_outliers=False)
    dftest = meli.feature_engineering(df_test, drop_outliers=False)

        
    
    print('# 4. EDA')
    meli.EDA(dftrain)


    print('# 5. Finding best model')
    print('NOTE: This might take a long time due to the number of models to train')
        
    ytrain = [1 if y=="new" else 0 for y in y_train]
    ytest = [1 if y=="new" else 0 for y in y_test]

    
    dfres = meli.find_best_model(dftrain, dftest, ytrain, ytest)



    print('# 6. Save lgbm model')
    model = meli.save_lgbm_model(dftrain, dftest, ytrain, ytest)
    
    
    
    print("# 7. Function to predict")
    """
    This is an example of how the model can be used to predict if the item is new or used
    from a dictionary input
    """
    d = X_test[1000]
    print(y_test[1000])
    meli.predict_meli(d)
    
    
    
    