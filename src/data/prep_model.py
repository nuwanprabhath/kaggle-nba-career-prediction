import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN 
from imblearn.combine import SMOTEENN
from imblearn.ensemble import RUSBoostClassifier, EasyEnsembleClassifier
from imblearn.under_sampling import RandomUnderSampler


# transform data by imputer and scaler
def transform(train, test, imputer, scaler):
       
    # fill np.nan 
    imputer = SimpleImputer(missing_values=np.nan, strategy=imputer)
    train = imputer.fit_transform(train)
    test = imputer.transform(test)
    
    # scaler = scaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    
    return train, test

# Find optimal number of cluster
def find_optimal_cluster(train, test):
    
    # Combine test & train to form all_X
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    all_X = train.append(test)

    # Find best number of clusters
    sil = []
    kmax = 12
    my_range=range(2,kmax+1)
    for i in my_range:
        kmeans = cluster.KMeans(n_clusters = i).fit(all_X)
        labels = kmeans.labels_
        sil.append(silhouette_score(all_X, labels, metric =  
        'correlation'))
        
    # Plot it, finding 2 is the ideal overall (all columns) cluster
    plt.plot(my_range, sil, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score by K')
    plt.show()

    return all_X


# Prepare training data for modelling - split and Resampling for imbalanced data
def prep_model_data(df_in, target, resample='', random_state=8):

    # Split randomly the dataset with random_state=8 into 2 different sets: data (80%) and test (20%)
    X_data, X_test, y_data, y_test = train_test_split(df_in, target, test_size=0.2, random_state=random_state, stratify=target)

    # Split the remaining data (80%) randomly with random_state=8 into 2 different sets: training (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=random_state, stratify=y_data) 
    
    #---------------------------------------------------------------------------------------
    # RESAMPLE if indicated
    
    # Original Training data
    if resample=='NO RESAMPLE' or resample=='':
        X_train_res, y_train_res = X_train, y_train
        X_val_res, y_val_res =  X_val, y_val
        X_test_res, y_test_res =  X_test, y_test
        #print(resample,'NO Resample -',features_no,'Features')
    
    # SMOTE - treat Imbalanced Target data after splitting Training & Test & Validation data
    if resample=='SMOTE':
        X_train_res, y_train_res = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
        X_val_res, y_val_res = SMOTE(random_state=random_state).fit_resample(X_val, y_val)
        X_test_res, y_test_res = SMOTE(random_state=random_state).fit_resample(X_test, y_test)
        #print(resample,'Resample -',features_no,'Features')

    # ADASYN - treat Imbalanced Target data after splitting Training & Test & Validation data
    if resample=='ADASYN':
        X_train_res, y_train_res = ADASYN(random_state=random_state).fit_resample(X_train, y_train)
        X_val_res, y_val_res = ADASYN(random_state=random_state).fit_resample(X_val, y_val)
        X_test_res, y_test_res = ADASYN(random_state=random_state).fit_resample(X_test, y_test)
        
    # SMOTEENN - treat Imbalanced Target data after splitting Training & Test & Validation data
    if resample=='SMOTEENN':
        X_train_res, y_train_res = SMOTEENN(random_state=random_state).fit_resample(X_train, y_train)
        X_val_res, y_val_res = SMOTEENN(random_state=random_state).fit_resample(X_val, y_val)
        X_test_res, y_test_res = SMOTEENN(random_state=random_state).fit_resample(X_test, y_test)
       
    # RANDOMUNDER - treat Imbalanced Target data after splitting Training & Test & Validation data
    if resample=='RANDOMUNDER':
        X_train_res, y_train_res = RandomUnderSampler(random_state=random_state).fit_resample(X_train, y_train)
        X_val_res, y_val_res = RandomUnderSampler(random_state=random_state).fit_resample(X_val, y_val)
        X_test_res, y_test_res = RandomUnderSampler(random_state=random_state).fit_resample(X_test, y_test)
              
    return X_train_res, y_train_res, X_val_res, y_val_res, X_test_res, y_test_res
