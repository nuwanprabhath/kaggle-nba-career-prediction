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
        df_in = df_in.drop(df_in[df_in[col] < 0].index)
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