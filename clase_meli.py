# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:47:31 2023

@author: Francisco Mena
"""

"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import os
import json
from unidecode import unidecode
import association_metrics as am
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import MutableMapping
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import lightgbm as lgb
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
import shap
import joblib

#%%

# Functions to flatten the nested dictionaries
def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            if v == []:
                v = np.nan
            yield new_key, v

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    return dict(_flatten_dict_gen(d, parent_key, sep))


#%%
class modeloMeli:


    @staticmethod
    def build_dataset():
        data = [json.loads(x) for x in open("MLA_100k_checked_v3.jsonlines")]
        target = lambda x: x.get("condition")
        N = -10000
        X_train = data[:N]
        X_test = data[N:]
        y_train = [target(x) for x in X_train]
        y_test = [target(x) for x in X_test]
        #for x in X_test:
        #    del x["condition"]
        return X_train, y_train, X_test, y_test
    
    
        

    @staticmethod
    def dict_to_df(x, i):
            
        """
        Function to convert a dictionary into a dataframe
        
        Parameters
        ----------
        x: dict
            dictionary with an item's data
        i: int
            number to use as index
            
        Returns
        -------
        The information in x as a dataframe
        
        """
        
        # 1. flatten the dictionary
        A = flatten_dict(x, sep = '_')
    
        
        # 2. correct features that are lists
        
        #If there is no non-meli payment method
        if type(A["non_mercado_pago_payment_methods"])!=list:        
            A["non_meli_payment__none"] = 1
        else:
            for method_dict in A["non_mercado_pago_payment_methods"]:
    
                new_key = "non_meli_payment__" + method_dict["description"]            
                A[new_key] = 1
    
        A.pop("non_mercado_pago_payment_methods")
    
    
        #Pictures
        if type(A["pictures"])!=list:        
            A["sum_pictures_quality"] = 0
            A["avg_picture_quality"] = 0
        else:
            pics = [p["size"].split('x') for p in A["pictures"]]
            A["sum_pictures_quality"] = sum([int(im[0])*int(im[1]) for im in  pics])
            A["avg_picture_quality"] = np.mean([int(im[0])*int(im[1]) for im in  pics])
    
    
        #Other lists
        for one_key, one_item in A.items():
            
            if type(one_item)==list:            
                A[one_key] = len(one_item)
    
    
        return pd.DataFrame(A, index = [i])
    
    
    #%%        
    @staticmethod
    def feature_engineering(df_train, drop_outliers=False):
        """
       Function to generate the new features that will be used during training
       
       Parameters
       ----------
       df_train: pandas dataframe
           Dataframe with the item's information
       drop_outliers: boolean
           Drop outliers in the digits feature, optional
           
       Returns
       -------
       Dataframe with the new features
       
       """
        
    
        cols_meli = [x for x in df_train.columns if "non_meli" in x]
        df_train[cols_meli] = df_train[cols_meli].fillna(0)
        
    
        # 1. Drop features with too many nans
        #feats_w_nans = list(df_train.isna().sum().loc[(df_train.isna().sum()/len(df_train)) >= 0.2].index)
        feats_w_nans = ['sub_status','deal_ids','shipping_methods','shipping_tags','shipping_dimensions','variations','attributes','tags',\
                        'parent_item_id','coverage_areas','official_store_id','differential_pricing','original_price','video_id',\
                        'catalog_product_id','subtitle','shipping_free_methods']
    
        dfaux = df_train.drop(feats_w_nans, axis = 1).copy()
    
        # 2. Drop features with too many similar values, or not useful
        feats2drop = ["seller_address_country_id","seller_address_country_name",\
                     "seller_address_state_id","seller_address_city_id",\
                      "site_id", "listing_source", "international_delivery_mode", \
                      "thumbnail", "secure_thumbnail", "permalink"
                      ] 
    
        dfaux.drop(feats2drop, axis = 1, inplace = True)
        
        
        
        
        
        # 2. Build features
        dfaux.set_index("id", inplace = True)
    
    
        dfaux["descriptions"].fillna(0, inplace = True)
    
        #date features
        dfaux["days_since_last_updated"] = (datetime.now( timezone.utc) - pd.to_datetime(dfaux["last_updated"])).dt.days
        dfaux["days_since_created"] = (datetime.now( timezone.utc) - pd.to_datetime(dfaux["date_created"])).dt.days
        dfaux["difference_days"] = dfaux["days_since_created"]-dfaux["days_since_last_updated"]
        dfaux.drop(["last_updated","date_created"], axis = 1, inplace = True)
        
    
        
        #picture features
        dfaux["quality_per_pic"] = dfaux["sum_pictures_quality"]/dfaux["pictures"]
        dfaux["pictures"].fillna(0, inplace = True)    
        dfaux["quality_per_pic"].fillna(0, inplace = True)    
        dfaux["sum_pictures_quality"].fillna(0, inplace = True)    
        dfaux["avg_picture_quality"].fillna(0, inplace = True)    
        
        
        #payment in USD 
        dfaux["currency_id"].value_counts()
        dfaux["currency_USD"] = [1 if x=="USD" else 0 for x in dfaux["currency_id"]]
        dfaux.drop(["currency_id"], axis = 1, inplace = True)
    
    
        #combinations of features
        dfaux["stop_minus_start"] = (dfaux["stop_time"] - dfaux["start_time"])
        dfaux["stop_div_start"] = dfaux["stop_time"]/dfaux["start_time"] - 1
    
        dfaux["difference_days"],dfaux["stop_div_start"]
        # dfaux["diff_days_over_stop_div_start"] = [x/y if y!=0 else 0 for x,y in zip(dfaux["difference_days"],dfaux["stop_div_start"])]
    
    
        dfaux["base_price_div_stop"] = dfaux["base_price"]/dfaux["stop_time"] * 1E10
    
    
        dfaux["base_price_div_days_since_updated"] = dfaux["base_price"]/dfaux["days_since_last_updated"] * 10
        
        
        dfaux["base_price_div_initial_quantity"]  = dfaux["base_price"]/dfaux["initial_quantity"]
        
        dfaux["total_money"] = dfaux["base_price"] * dfaux["initial_quantity"]
        
        # dfaux["sold_over_initial_quantity"] = [x/y if y!=0 else 0 for x,y in zip(dfaux["sold_quantity"],dfaux["initial_quantity"])]
                                           
    
    
    
        # Seller frequency
        def freq_seller_id(s):
            if s == 1: 
                return "sold_once"
            elif (s > 1) & (s < 4):
                return "rare_sells"
            else:
                return "frequent_seller"
        
        freq_seller = dfaux["seller_id"].value_counts().apply(freq_seller_id)
        freq_seller = freq_seller.to_frame("seller_frequency") 
        freq_seller = freq_seller.reset_index().rename(columns={"index":"seller_id"})
        
        dfaux = pd.merge(dfaux, freq_seller, how = "left", left_on=["seller_id"], right_on = ["seller_id"])
        dfaux.drop("seller_id", axis = 1, inplace = True)
    
        # Category frequency
        def freq_category_id(s):
            if s == 1: 
                return "unique_category"
            elif (s > 1) & (s <= 6):
                return "rare_category"
            else:
                return "common_category"
        
        freq_catego = dfaux["category_id"].value_counts().apply(freq_category_id)
        freq_catego = freq_catego.to_frame("category_frequency") 
        freq_catego = freq_catego.reset_index().rename(columns={"index":"category_id"})
        
        dfaux = pd.merge(dfaux, freq_catego, how = "left", left_on=["category_id"], right_on = ["category_id"])
        dfaux.drop("category_id", axis = 1, inplace = True)
        
        
        # State frequency - create categories  
        #list(dfaux["seller_address_state_name"].value_counts().loc[dfaux["seller_address_state_name"].value_counts()<200].index)
        rare_states = ['San Juan','Salta','Misiones','Río Negro','Corrientes','Neuquén','La Pampa','Chaco','San Luis','Jujuy',
         'Formosa','Santiago del Estero','Santa Cruz','Catamarca','La Rioja','Tierra del Fuego','']
        
        dfaux["seller_address_state_name"] = ["rare_state" if x in rare_states else x for x in dfaux["seller_address_state_name"]]
            
    
        # City frequency - create categories
        common_cities = ['CABA','Palermo','Buenos Aires','Capital Federal','Mataderos','Caballito','Villa Crespo','capital federal']
        
        dfaux["seller_address_city_name"] = ["other_city" if x not in common_cities else x for x in dfaux["seller_address_city_name"]]
        
        
        # Warranty
        import re
        import nltk
        from nltk.corpus import stopwords
        nltk.download('punkt')
        nltk.download('stopwords')
        stopword = stopwords.words('spanish')
    
        def remove_stopwords(text):
            text = [word for word in text.split(' ') if word not in stopword]
            text = ' '.join(text)
            text = text.translate({ord(i): '' for i in '-.!,/#+*'})
            
            return text
    
        
        dfaux["warranty"].fillna("no_warranty_info", inplace = True)
        dfaux["warranty"] = [unidecode(x.lower()) for x in dfaux["warranty"]]
        dfaux["warranty"] = dfaux["warranty"].apply(lambda x: remove_stopwords(x) )
    
    
        """ This commented code can be used for finding the most frequent words in the text
        # textnew = " ".join(dfaux.loc[ dfaux["condition"] == "new", "warranty"].values)
        # textused = " ".join(dfaux.loc[ dfaux["condition"] == "used", "warranty"].values)
    
        # allWords = nltk.tokenize.word_tokenize(textnew)
        # allWordDist = nltk.FreqDist(w.lower() for w in allWords)
        # #print( allWordDist.most_common(40) )
        
        # allWords = nltk.tokenize.word_tokenize(textused)
        # allWordDist = nltk.FreqDist(w.lower() for w in allWords)
        # #print( allWordDist.most_common(40) )
        """
    
        def garantia(text):
            "frequency of words"
            si_garantia = ["si","fabrica","ano","mes","dia","devol", "oficial", "defect", "falla", "nuev"]
            no_garantia = ["sin", "usar", "usad", "calific", "coment", "reput", "antig", "vinil", "origin", "estad"]
            
            res = text
            if any( word in text for word in si_garantia):
                res = "yes_warranty"
            elif any( word in text for word in no_garantia):
                res = "no_warranty"
            elif text != "no_warranty_info":
                res = "unsure_warranty"
            return res
    
    
        dfaux["garantia"] = dfaux["warranty"].apply(garantia)
    
        #If the text has a digit    
        def number_getter(s):
            valor = 0
            
            try:
                re.findall(r'\d+' , s)[0]
                valor = 1
            except:
                pass
            return valor
            
        dfaux["digits"] = dfaux["warranty"].apply(number_getter)
    
        if drop_outliers:
            idx2drop = dfaux.loc[ (dfaux["digits"]==1) & (dfaux["condition"]=="used") ].index
            dfaux = dfaux.drop(idx2drop)
    
        dfaux.drop("warranty", axis = 1, inplace = True)
        
    
    
    
        # Title - similar analysis as with warranty
        dfaux["title"].fillna("no_title_info", inplace = True)
        dfaux["title"] = [unidecode(x.lower()) for x in dfaux["title"]]
        dfaux["title"] = dfaux["title"].apply(lambda x: remove_stopwords(x) )
    
    
        # textnew = " ".join(dfaux.loc[ dfaux["condition"] == "new", "title"].values)
        # textused = " ".join(dfaux.loc[ dfaux["condition"] == "used", "title"].values)
    
        # allWords = nltk.tokenize.word_tokenize(textnew)
        # allWordDist = nltk.FreqDist(w.lower() for w in allWords)
        # # print( allWordDist.most_common(40) )
        
        # allWords = nltk.tokenize.word_tokenize(textused)
        # allWordDist = nltk.FreqDist(w.lower() for w in allWords)
        # # print( allWordDist.most_common(40) )
    
    
        def titulo(text):
    
            title_new = ["nuev", "original", "kit"]
            title_used = ["vinilo", "antigu", "revista", "excelente", "usad", "impecable", "digital"]
            
            res = text
            if any( word in text for word in title_new):
                res = "title_new"
            elif any( word in text for word in title_used):
                res = "title_used"
            else:
                res = "other_title"
    
            return res
    
    
        dfaux["titulo"] = dfaux["title"].apply(titulo)
        dfaux.drop(["title"], axis = 1, inplace = True)
    
            
        #check no NaNs
        assert dfaux.isna().sum().sum() == 0
    
    
        return dfaux
    
    #%%    
    @staticmethod
    def EDA(dfeatsog):
        """
        Function to plot different features vs the items' condition
        Parameters
        ----------        
        dfeatsog: pandas dataframe
            Dataframe with the items' information
        
        Returns
        -------
        Plots describing patterns in the items' condition
        
        """
        
        dfeats = dfeatsog.copy()
        dfeats["new"] = np.where(dfeats["condition"]=="new", 1, 0)
    
        folder2save = os.path.join( os.getcwd(), "Plots")
            
        if not os.path.exists( os.path.join( os.getcwd(), "Plots")):
            os.mkdir( os.path.join( os.getcwd(), "Plots") )
    
        # New items tend to be more expensive
        sns.boxplot(data = dfeats, x = "condition", y = "price")
        plt.ylim(0, 4000)
        plt.savefig(folder2save + "/Boxplot_price.png", dpi = 100, bbox_inches = "tight")        
        plt.close()
        
        # Max price is very large, might be outlier
        print(dfeats["price"].max())
    
        # Not much of a difference on stop or start time
        g = sns.boxplot(data = dfeats, x = "condition", y = "start_time")
        g.set_yscale("log")
        plt.ylim(1.43E12, 1.45E12)
        plt.savefig(folder2save + "/Boxplot_start_time.png", dpi = 100, bbox_inches = "tight")        
        plt.close()
    
        # Pearson's correlation
        sns.heatmap(dfeats.corr(), cmap="YlGnBu")
        plt.savefig(folder2save + "/Pearson_corr.png", dpi = 100, bbox_inches = "tight")        
        plt.close()
        """
        Pearson corr: There are some understandably high correlations, for instance, available_quantity and initial_quantity,
        or price and base_price. There are some interesting negative correlations, like currency_USD and accepts_mercadopago,
        days since created or last updated and start_time
    
        """
    
        # Cramer's V coefficient 
        # To understand the strength of the relationship between two categorical variables
        dfeats = dfeats.apply(lambda x: x.astype("category") if x.dtype == "O" else x)
    
        # Initialize a CamresV object using you pandas.DataFrame
        dfeats.dtypes
        cat_cols = ["shipping_mode", "listing_type_id","buying_mode","status","seller_frequency","category_frequency"
                    ,"condition"]
        cramersv = am.CramersV(dfeats[cat_cols])
        dfcramer = cramersv.fit()
        # will return a pairwise matrix filled with Cramer's V, where columns and index are 
        # the categorical variables of the passed pandas.DataFrame
        sns.heatmap(dfcramer, cmap="YlGnBu")
        plt.savefig(folder2save + "/CramersV_corr.png", dpi = 100, bbox_inches = "tight")        
        plt.close()
        """
        Cramer's V: The condition of the item is most strongly correlated with listing_type_id,
        followed by seller_frequency and buying mode. The other categories are not so strongly correlated
        """
        
        dfeats.drop(["new"], axis = 1, inplace = True)
    
        return
    
    #%%
    @staticmethod
    def find_best_model(dftrain, dftest, ytrain, ytest):
        """
        Function that trains trains several different models, and finds the one with best accuracy
        Parameters
        ----------
        df_train : pandas DataFrame
            Dataframe with the training set
        df_test : pandas DataFrame
            Dataframe with the testing set
        ytrain : list
            training label
        ytest : list
            testing label
            
        Returns
        -------
        DataFrame with the metrics for the models trained
        """
        
        # One-Hot Encoding
        # Convert to category
        dftrain= dftrain.apply(lambda x: x.astype("category") if x.dtype == "O" else x)
        dftest = dftest.apply(lambda x: x.astype("category") if x.dtype == "O" else x)
        categories = ['seller_address_state_name', 'seller_address_city_name', 'shipping_mode', 'listing_type_id',\
                      'buying_mode', 'status','seller_frequency', 'category_frequency', 'garantia',"titulo"]
        
        dftrain.drop(["condition"], axis = 1, inplace = True)
        dftest.drop(["condition"], axis = 1, inplace = True)
    
        
    
        dftrain2 = dftrain.copy()
        dftest2 = dftest.copy()
        
        dic_ohes = {}
        for cat in categories:
            
            
            ohe_one = OneHotEncoder()
            A = pd.DataFrame(  ohe_one.fit_transform(dftrain[[cat]]).toarray(), columns = ohe_one.categories_[0] , index = dftrain.index)
            dftrain = dftrain.merge(A[ohe_one.categories_[0]], left_index = True, right_index = True)
            
            B = pd.DataFrame(  ohe_one.transform(dftest[[cat]]).toarray(), columns = ohe_one.categories_[0] , index = dftest.index)
            dftest = dftest.merge(B[ohe_one.categories_[0]], left_index = True, right_index = True)
            
            dic_ohes[cat] = ohe_one
        
        dftrain.drop(categories, axis = 1, inplace = True)
        dftest.drop(categories, axis = 1, inplace = True)
        
        # Order
        cols = dftrain.columns
        dftest = dftest[cols]
        cols2 = dftrain2.columns
        dftest2 = dftest2[cols2]
        
        # Scale
        scale = StandardScaler()
    
        Xtrain_scaled = scale.fit_transform(dftrain)
        Xtest_scaled = scale.transform(dftest)
    
    
        # Train and test models
        """
        Baseline
        """
        from sklearn.dummy import DummyClassifier
        clf_dum = DummyClassifier()
        clf_dum.fit(Xtrain_scaled, ytrain)
        ypred = clf_dum.predict(Xtest_scaled)
        
        print("accuracy: " + str(accuracy_score(ytest, ypred)))
        
        
        
        """
        Let's start with something simple. A logistic regression with search of parameter C
        """        
    
        clf_lr = LogisticRegressionCV(cv = 5, scoring = "accuracy", n_jobs=-1, random_state=2077)
        
        clf_lr.fit(Xtrain_scaled, ytrain)
        
        ypred = clf_lr.predict(Xtest_scaled)
        
        print("Logistic Regression CV score")
        print("accuracy: " + str(accuracy_score(ytest, ypred)))
        print("roc_auc: " + str(roc_auc_score(ytest, ypred)))
        print("confussion matrix: " + str( confusion_matrix(ytest, ypred) )    )
        print(classification_report(ytest, ypred))
        
        
        df_logreg = pd.DataFrame({"acc":np.round(accuracy_score(ytest, ypred),2),
                      "roc_auc": np.round(roc_auc_score(ytest, ypred),2),
                          }, index = ["log_reg"])
        
        
        
        """
        Random Forest
        """
            
        
        clf_rf = RandomForestClassifier(random_state=2077, n_jobs=-1, class_weight = "balanced")    
        clf_rf.fit(Xtrain_scaled, ytrain)
        ypred = clf_rf.predict(Xtest_scaled)
        
        print("Random Forest score")
        print("accuracy: " + str(np.round(accuracy_score(ytest, ypred),3)  ))
        print("roc_auc: " + str(roc_auc_score(ytest, ypred)))
        print("confussion_matrix: " + str( confusion_matrix(ytest, ypred) )    )
        print(classification_report(ytest, ypred))
        
        df_rf = pd.DataFrame({"acc":np.round(accuracy_score(ytest, ypred),3),
                      "roc_auc": np.round(roc_auc_score(ytest, ypred),3),
                          }, index = ["RF"])
        
    
        """ feature importance RF """
        print("RF feature importance")
        feature_list = dftrain.columns
        f_importance = list(clf_rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, f_importance)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]    
    
    
    
        """
        Linear SVC
        """
        
        clf_svc = LinearSVC(random_state=2077, dual = False, class_weight = "balanced")
        
        clf_svc.fit(Xtrain_scaled, ytrain)
        ypred = clf_svc.predict(Xtest_scaled)
        
        print("SVC score")
        print("accuracy: " + str(np.round(accuracy_score(ytest, ypred),2)  ))
        print("roc_auc: " + str(roc_auc_score(ytest, ypred)))
        print("confussion_matrix: " + str( confusion_matrix(ytest, ypred) )    )
        print(classification_report(ytest, ypred))
        
        df_svc = pd.DataFrame({"acc":np.round(accuracy_score(ytest, ypred),2),
                      "roc_auc": np.round(roc_auc_score(ytest, ypred),2),
                          }, index = ["SVC"])
        
    
        """
        LightGBM
        """
    
        # LightGBM    
        # create dataset for lightgbm
        dftrain3 = dftrain2.drop(["avg_picture_quality"], axis = 1).copy()
        dftest3 = dftest2.drop(["avg_picture_quality"], axis = 1).copy()

        
        
        lgb_train = lgb.Dataset(dftrain3, label = ytrain)
        lgb_eval = lgb.Dataset(dftest3, label = ytest, reference=lgb_train)
    
        # specify your configurations as a dict
        params = {
            'n_estimators':1000,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 100,
            'learning_rate': 0.06,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 10,
            'verbose': 0,
            'random_state':42
        }    
            
    
        print('Starting training lgb...')
        # feature_name and categorical_feature
        gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval ,  # eval training data
                    feature_name=list(dftrain3.columns),
                    #callbacks=[lgb.early_stopping(stopping_rounds=20)],
                    categorical_feature=['seller_address_state_name', 'seller_address_city_name',
                           'shipping_mode', 'listing_type_id', 'buying_mode', 'status',
                           'seller_frequency', 'category_frequency', "garantia"]
                    )    
        
    
        """ LightGBM returns the probability of belonging to a class. 
        Since there is a slight unbalance in the classes, we use the class
        proportions to choose one of the two classes"""
        ypred = gbm.predict(dftest3)
        ypred = [1 if y>0.55 else 0 for y in ypred]
    
    
        print("lightgbm score")
        print("accuracy: " + str(100 * np.round(accuracy_score(ytest, ypred),3)  ))
        print("roc_auc: " + str(100 * roc_auc_score(ytest, ypred)))
        print("confussion_matrix: " + str( confusion_matrix(ytest, ypred) )    )
        print(classification_report(ytest, ypred))
        
        df_lgb = pd.DataFrame({"acc":100 * np.round(accuracy_score(ytest, ypred),3),
                      "roc_auc": 100 * np.round(roc_auc_score(ytest, ypred),3),
                          }, index = ["LGB"])
        
        
        
        
        folder2save = os.path.join( os.getcwd(), "Plots")    
        lgb.plot_importance(gbm, figsize=(10,10))
        plt.savefig(folder2save + "/FeatureImportance_LGB.png", dpi = 100, bbox_inches = "tight")        
        plt.close()
        
    
        """ Explain lgb model """
        
        explainer = shap.TreeExplainer(gbm)
        shap_values = explainer.shap_values(dftest2)
        shap.summary_plot(shap_values, dftest2)
        plt.savefig(folder2save + "/SHAP_LGB.png", dpi = 100, bbox_inches = "tight")        
        
        
    
        """XGB"""
        
        bst = XGBClassifier(n_estimators=1000, max_depth=100, learning_rate=0.01, objective='binary:logistic', n_jobs = -1, random_state=42)
    
        bst.fit(Xtrain_scaled, ytrain, eval_set = [(Xtest_scaled,ytest)] )
    
        ypred = bst.predict(Xtest_scaled)
        
        print("XGB score")
        print("accuracy: " + str(np.round(accuracy_score(ytest, ypred),2)  ))
        print("roc_auc: " + str(roc_auc_score(ytest, ypred)))
        print("CM: " + str( confusion_matrix(ytest, ypred) )    )
        print(classification_report(ytest, ypred))
    
    
        df_xgb = pd.DataFrame({"acc":np.round(accuracy_score(ytest, ypred),2),
                      "roc_auc": np.round(roc_auc_score(ytest, ypred),2),
                          }, index = ["XGB"])
    
        
        """ Tensorflow """        
    
        tf.keras.backend.clear_session()
        
        model = keras.Sequential()
        
        model.add( keras.layers.Input(shape = Xtrain_scaled.shape[1:]))
        model.add( keras.layers.Dense(units = 80, activation = 'relu', kernel_initializer = 'he_normal'))
        model.add( keras.layers.Dropout(0.1))
        model.add( keras.layers.Dense(units = 40, activation = 'relu', kernel_initializer = 'he_normal'))
        model.add( keras.layers.Dropout(0.1))
        model.add( keras.layers.Dense(units = 20, activation = 'relu', kernel_initializer = 'he_normal'))
        model.add( keras.layers.Dropout(0.1))
        model.add( keras.layers.Dense(units = 10, activation = 'relu', kernel_initializer = 'he_normal'))
        model.add( keras.layers.Dropout(0.1))
        model.add( keras.layers.Dense(units = 1, activation = 'sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        print(model.summary())    
    
    
        early_stop = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)
    
        history = model.fit(Xtrain_scaled, np.array(ytrain), 
                            batch_size=8, 
                            epochs=50, 
                            verbose=1, 
                            validation_split=0.2,
                            callbacks=[early_stop]
                            )
    
    
        score = model.evaluate(Xtest_scaled, np.array(ytest), verbose=1)
    
        ypred = model.predict(Xtest_scaled)
        ypred = [1 if x[0]>0.5 else 0 for x in ypred]
    
        
        print("TF score")
        print("accuracy: " + str(np.round(accuracy_score(ytest, ypred),2)  ))
        print("roc_auc: " + str(roc_auc_score(ytest, ypred)))
        print("confussion_matrix: " + str( confusion_matrix(ytest, ypred) )    )
        print(classification_report(ytest, ypred))
    
        df_tf = pd.DataFrame({"acc":np.round(accuracy_score(ytest, ypred),2),
                      "roc_auc": np.round(roc_auc_score(ytest, ypred),2),
                          }, index = ["TF"])
    
    
        return pd.concat([df_logreg,df_rf,df_svc,df_lgb,df_xgb,df_tf])

#%%
    @staticmethod
    def save_lgbm_model(dftrain, dftest, ytrain, ytest):
        """
        Function that trains trains a LightGBM model and saves it with joblib
        Parameters
        ----------
        dftrain : pandas DataFrame
            Dataframe with the training set
        dftest : pandas DataFrame
            Dataframe with the testing set
        ytrain : list
            training label
        ytest : list
            testing label
            
        Returns
        -------
        LigthGBM model
        """

        dftrain= dftrain.apply(lambda x: x.astype("category") if x.dtype == "O" else x)
        dftest = dftest.apply(lambda x: x.astype("category") if x.dtype == "O" else x)
        
        if "condition" in dftrain.columns:
            dftrain.drop(["condition"], axis = 1, inplace = True)
        if "condition" in dftest.columns:
            dftest.drop(["condition"], axis = 1, inplace = True)
            
    
        dftrain2 = dftrain.copy()
        dftest2 = dftest.copy()
    
        cols2 = dftrain2.columns
        dftest2 = dftest2[cols2]
    
        dftrain3 = dftrain2.drop(["avg_picture_quality"], axis = 1).copy()
        dftest3 = dftest2.drop(["avg_picture_quality"], axis = 1).copy()
        
        
        import lightgbm as lgb
        lgb_train = lgb.Dataset(dftrain3, label = ytrain)
        lgb_eval = lgb.Dataset(dftest3, label = ytest, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            'n_estimators':1000,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 100,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 10,
            'verbose': 0,
            'random_state':42
        }    
            

        print('Starting training...')
        # feature_name and categorical_feature
        gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval ,  # eval training data
                    feature_name=list(dftrain3.columns),
                    #callbacks=[lgb.early_stopping(stopping_rounds=20)],
                    categorical_feature=['seller_address_state_name', 'seller_address_city_name',
                           'shipping_mode', 'listing_type_id', 'buying_mode', 'status',
                           'seller_frequency', 'category_frequency', "garantia", "titulo"]
                    )    
        

        ypred = gbm.predict(dftest3)
        ypred = [1 if y>0.54 else 0 for y in ypred]


        print("lightgbm score")
        print("accuracy: " + str(100 * np.round(accuracy_score(ytest, ypred),3)  ))
        print("roc_auc: " + str(100 * roc_auc_score(ytest, ypred)))
        print("confussion_matrix: " + str( confusion_matrix(ytest, ypred) )    )
        print(classification_report(ytest, ypred))


        # Save model
        folder4model = os.path.join( os.getcwd(), "Models")
            
        if not os.path.exists( folder4model ):
            os.mkdir( folder4model )

        joblib.dump(gbm, os.path.join( folder4model,"meli_lgbm_classifier.joblib"))
        
        return gbm
    
    
#%%    
    
    @staticmethod 
    def predict_meli(x:dict):
        """
        Function that, given a dictionary with an item's information, 
        predicts if the item is new or used.
        Parameters
        ----------
        x : dict
            Dictionary with the item's information
        Returns
        -------
        The class of the item (new or used)
        """
        
        
        #convert dictionary to dataframe
        meli = modeloMeli()
        dfaux = meli.dict_to_df(x, 0)
 

        # Verify that all columns are present
        feature_columns = ['seller_address_country_name', 'seller_address_country_id','seller_address_state_name', 'seller_address_state_id',
               'seller_address_city_name', 'seller_address_city_id', 'warranty','sub_status', 'condition', 'deal_ids', 'base_price',
               'shipping_local_pick_up', 'shipping_methods', 'shipping_tags','shipping_free_shipping', 'shipping_mode', 'shipping_dimensions',
               'seller_id', 'variations', 'site_id', 'listing_type_id', 'price','attributes', 'buying_mode', 'tags', 'listing_source', 'parent_item_id',
               'coverage_areas', 'category_id', 'descriptions', 'last_updated','international_delivery_mode', 'pictures', 'id', 'official_store_id',
               'differential_pricing', 'accepts_mercadopago', 'original_price','currency_id', 'thumbnail', 'title', 'automatic_relist', 'date_created',
               'secure_thumbnail', 'stop_time', 'status', 'video_id','catalog_product_id', 'subtitle', 'initial_quantity', 'start_time',
               'permalink', 'sold_quantity', 'available_quantity','non_meli_payment__Transferencia bancaria',
               'non_meli_payment__Acordar con el comprador','non_meli_payment__Efectivo', 'sum_pictures_quality',
               'avg_picture_quality', 'non_meli_payment__Tarjeta de crédito','non_meli_payment__none', 'shipping_free_methods',
               'non_meli_payment__MasterCard', 'non_meli_payment__Mastercard Maestro','non_meli_payment__Visa Electron', 'non_meli_payment__Contra reembolso',
               'non_meli_payment__Visa', 'non_meli_payment__Diners','non_meli_payment__American Express', 'non_meli_payment__Giro postal',
               'non_meli_payment__MercadoPago','non_meli_payment__Cheque certificado']
        
        for col in feature_columns:
            if col not in dfaux.columns:
                
                dfaux[col] = np.nan

           
        #features
        df = meli.feature_engineering(dfaux)
    
        df = df.apply(lambda x: x.astype("category") if x.dtype == "O" else x)
            
        
        if "condition" in df.columns:
            df.drop(["condition"], axis = 1, inplace = True)
            
    
        df.columns = [x.replace(' ','_') for x in df.columns]

        
        #Load model    
        folder4model = os.path.join( os.getcwd(), "Models")
        lgb_model = joblib.load( os.path.join( folder4model,"meli_lgbm_classifier.joblib") )

        #select features 
        df = df[lgb_model.feature_name()]

        #predict
        ypred_proba = lgb_model.predict(df)
        
        ypred = np.where(ypred_proba > 0.54, "new", "used")
        
        return "The item is " + ypred[0]
    
    #%%
    #%%
    #%%
    
