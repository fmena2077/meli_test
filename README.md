# Mercado Libre's Data Science Code Exercise
Author: Francisco Mena

This is the repository for Mercado Libre's Data Science Code Exercise


## Content

* clase_meli.py: class with the functions for training and implementing a machine learning model
that predicts if an item is used or new.


* main_script.py: Script that shows how data is loaded, data cleaning, feature engineering, EDA,
how the best model is chosen, and given an input dictionary, the model can predict if the item is used or new.
  

* Models/meli_lgbm_classifier.joblib: Joblib of the best model found in main_script.py


* Plots: Folder with the plots created in the EDA
  

* MELI_test_FranciscoMena.ipynb: Jupyter that shows the same steps for training the ML model

NOTE: The MLA_100k_checked_v3.jsonlines file with the data is not available in this repository,
hence one has to include it first for the scripts to work.