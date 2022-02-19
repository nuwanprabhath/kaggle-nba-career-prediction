import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # To normalise columns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector # For feature selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

def split_and_normalize(train_file, processed_folder):
    """Split train data into multiple files with stratify parameter as yes"""
    train_df = pd.read_csv(train_file)
    X_df = train_df.loc[:, 'GP':'TOV'] # Selecting data fields so Id column is not getting normalized
    Id_df = train_df.loc[:, 'Id']
    y = train_df.loc[:, 'TARGET_5Yrs']
    
    print("Original train shape: {}".format(train_df.shape))
    
    scaler = MinMaxScaler() # Min max scaler
    X = scaler.fit_transform(X_df) # Scaling train set
    X = pd.DataFrame(np.squeeze(X), columns=X_df.columns) # Convert X back to pd data frame 
    X = pd.concat([Id_df, X], axis=1)

    print("Concat shape: {}".format(X.shape))
    
    # Source of this code section below: https://towardsdatascience.com/how-to-split-data-into-three-sets-train-validation-and-test-and-why-e50d22d3e54c
    # split the data in training and remaining dataset. 80:10:10 for train:valid:test
    X_train, X_rem, y_train, y_rem  = train_test_split(X, y, train_size=0.8, shuffle = True, stratify=y, random_state = 8)
    
    # Valid and test size to be equal (10% each of overall data), valid_size=0.5 (that is 50% of remaining data)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, shuffle = True, stratify=y_rem, random_state = 8)

    # Returned y values are Pandas Series and converting them to DataFrame
    y_train = y_train.to_frame()
    y_test = y_test.to_frame()
    y_valid = y_valid.to_frame()
    
    X_train.drop('Id', axis=1, inplace=True) # In training set, Id is not required. Therefore removing it

    X_train.to_csv(processed_folder+'/X_train.csv', index = False)
    y_train.to_csv(processed_folder+'/y_train.csv', index = False)
    
    X_test.to_csv(processed_folder+'/X_test.csv', index = False)
    y_test.to_csv(processed_folder+'/y_test.csv', index = False)
    
    X_valid.to_csv(processed_folder+'/X_valid.csv', index = False)
    y_valid.to_csv(processed_folder+'/y_valid.csv', index = False)

    print("Files written to: {}".format(processed_folder))
    
    print("X_train shape: {}".format(X_train.shape)) # Columns from Id to TOV
    print("y_train shape: {}".format(y_train.shape)) # Just TARGET_5Yrs column

    print("X_test shape: {}".format(X_test.shape))   # Columns from Id to TOV
    print("y_test shape: {}".format(y_test.shape))   # Just TARGET_5Yrs column
    
    print("X_valid shape: {}".format(X_valid.shape)) # Columns from Id to TOV
    print("y_valid shape: {}".format(y_valid.shape)) # Just TARGET_5Yrs column

    return X_train, y_train, X_test, y_test, X_valid, y_valid


def load_kaggle_train_and_test_data(train_file, test_file):
    """ Load  Kaggle data and normalize columns"""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    train_set = train_df.loc[:, 'GP':'TOV']     # Selecting all columns except Id and TARGET_5Yrs as training set to normalize
    train_set_Id = train_df.loc[:, 'Id']        # Ids of train set 
    target_set = train_df.loc[:, 'TARGET_5Yrs'] # Selecting TARGET_5Yrs as the target

    test_set = test_df.loc[:, 'GP':'TOV']       # Test set except Id
    test_set_Id = test_df.loc[:, 'Id']          # Ids of the test set

    scaler = MinMaxScaler() # Min max scaler

    train_norm = scaler.fit_transform(train_set) # Scaling train set
    test_norm = scaler.fit_transform(test_set)   # Scaling test

    train_set = pd.DataFrame(np.squeeze(train_norm), columns=train_set.columns) # Converting numpy array result to pandas dataframe
    test_set = pd.DataFrame(np.squeeze(test_norm), columns=test_set.columns)

    test_set = pd.concat([test_set_Id, test_set], axis=1)  # Append respective Ids. Ids are required to relate the prediction to an Id

    print("X_train shape: {}".format(train_set.shape))
    print("y_train shape: {}".format(target_set.shape))
    print("X_test shape: {}".format(test_set.shape))   
    
    return train_set, target_set, test_set


def sequential_feature_selection(path, num_features):
    """Select features based on the given train test and return selected features
    path: Path with the train features and labels
    num_features: Number of features to select
    """
    train_df = pd.read_csv(path)
    train_set = train_df.loc[:, 'GP':'TOV']

    scaler = MinMaxScaler() # Min max scaler
    train_norm = scaler.fit_transform(train_set) # Scaling train set
    X = pd.DataFrame(np.squeeze(train_norm), columns=train_set.columns) # Converting numpy array result to pandas dataframe
    y = train_df.loc[:, 'TARGET_5Yrs'] # Selecting TARGET_5Yrs as the target

    knn = KNeighborsClassifier(n_neighbors=3)
    sfs = SequentialFeatureSelector(knn, n_features_to_select=num_features)
    sfs.fit(X, y)
    features = sfs.get_feature_names_out()
    print(features)
    return features.tolist() # Converting numpy array to a normal array



