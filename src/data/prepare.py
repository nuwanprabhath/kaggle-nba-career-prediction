# A collection of helpful data preparation functions

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def remove_invalid_rows(df_in, cols_in):
    """Drop observations where feature value is negative

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe containing features to be interrogated

    cols_in : list
        List of columns to be interrogated

    Returns
    -------
    df_in : pd.DataFrame
    """

    for col in cols_in:
        df_in = df_in.drop(df_in[df_in[col] < 0].index, inplace=True)
    return df_in

def replace_negatives(df_in, cols_in):
    """Replace negative values with feaure mean value

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe containing features to be interrogated

    cols_in : list
        List of columns to be interrogated

    Returns
    -------
    df_in : pd.DataFrame
    """    
    for col in cols_in:
        df_in[col] = np.where((df_in[col] < 0), np.nan, df_in[col])
        meanVal = df_in[col].mean()
        df_in[col] = df_in[col].fillna(meanVal)
    return df_in

def drop_features(df_in, cols_in):
    """Drop features in column list

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe containing features to be interrogated

    cols_in : list
        List of columns to be dropped

    Returns
    -------
    df_in : pd.DataFrame
    """

    for col in cols_in:
        df_in.drop([col], axis=1, inplace=True)
    return df_in

def scale_features(df_in, scaler, target_col=None):
    """Split sets randomly

    Parameters
    ----------
    df_in : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    scaler : float
        Scaling processor to fit to features

    Returns
    -------
    X : pd.DataFrame
        Dataframe containing scaled data
    """

    X = df_in.copy()
    if target_col is not None:
        y = X.pop(target_col)

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(np.squeeze(X_scaled), columns=X.columns)
    if target_col is not None:
        X = pd.concat([X_scaled, y], axis=1)
    else:
        X = X_scaled

    return X

def split_sets(df_in, target_col):
    """Split sets randomly

    Parameters
    ----------
    df_in : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    """

    X = df_in.copy()
    y = X.pop(target_col)

    X_train, X_val, y_train, y_val  = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=8)
    return X_train, y_train, X_val, y_val

# Modified function sourced from ADV DSI Lab Solution to accept set name to be used in filename when writing NP array to filesystem
# Author: Anthony So

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, set_name=None, path='../data/processed/'):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')
    setname : str
        Set Name to be appended to file name during write

    Returns
    -------
    """

    if set_name is not None:
       l_setname = '_' + set_name
    else:
       l_setname = ''

    if X_train is not None:
      np.save(f'{path}X_train{l_setname}', X_train)
    if X_val is not None:
      np.save(f'{path}X_val{l_setname}',   X_val)
    if X_test is not None:
      np.save(f'{path}X_test{l_setname}',  X_test)
    if y_train is not None:
      np.save(f'{path}y_train{l_setname}', y_train)
    if y_val is not None:
      np.save(f'{path}y_val{l_setname}',   y_val)
    if y_test is not None:
      np.save(f'{path}y_test{l_setname}',  y_test)

# Modified function sourced from ADV DSI Lab Solution to accept set name to be used in filename when reading NP array from filesystem
# Author: Anthony So

def load_sets(set_name=None, val=False, path='../data/processed/'):

    """Load the different locally save sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    import os.path

    if set_name is not None:
       l_setname = '_' + set_name
    else:
       l_setname = ''

    X_train = np.load(f'{path}X_train{l_setname}.npy') if os.path.isfile(f'{path}X_train{l_setname}.npy') else None
    X_val   = np.load(f'{path}X_val{l_setname}.npy'  ) if os.path.isfile(f'{path}X_val{l_setname}.npy')   else None
    X_test  = np.load(f'{path}X_test{l_setname}.npy' ) if os.path.isfile(f'{path}X_test{l_setname}.npy')  else None
    y_train = np.load(f'{path}y_train{l_setname}.npy') if os.path.isfile(f'{path}y_train{l_setname}.npy') else None
    y_val   = np.load(f'{path}y_val{l_setname}.npy'  ) if os.path.isfile(f'{path}y_val{l_setname}.npy')   else None
    y_test  = np.load(f'{path}y_test{l_setname}.npy' ) if os.path.isfile(f'{path}y_test{l_setname}.npy')  else None
    
    return X_train, y_train, X_val, y_val, X_test, y_test