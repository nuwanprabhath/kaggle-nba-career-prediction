# A collection of helpful data exploration functions

import pandas as pd

def print_na_info(df_in):
    """Print information about null values in data set

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe containing features to be interrogated

    Returns
    -------
    N/A
    """

    isna_sum_sorted = df_in.isna().sum().sort_values(ascending=False)
    len_df_in = len(df_in)
    df_feat_missing = pd.DataFrame ({
                      'feature': isna_sum_sorted.index,
                      'null_value_count': isna_sum_sorted.values,
                      'null_value_percent': 100 * isna_sum_sorted.values / len_df_in
                      })
    df_feat_missing.drop(df_feat_missing[df_feat_missing['null_value_count']== 0].index, inplace=True)
    df_feat_missing.reset_index(inplace=True, drop=True)
    display(df_feat_missing)

def print_negative_info(df_in):
    """Print information about negative values in data set

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe containing features to be interrogated

    Returns
    -------
    N/A
    """

    neg_sum_sorted = (df_in < 0).sum().sort_values(ascending=False)
    len_df_in = len(df_in)
    df_feat_neg = pd.DataFrame ({
                  'feature': neg_sum_sorted.index,
                  'negative_value_count': neg_sum_sorted.values,
                  'negative_value_percent': 100 * neg_sum_sorted.values / len_df_in
                  })
    df_feat_neg.drop(df_feat_neg[df_feat_neg['negative_value_count']== 0].index, inplace=True)
    df_feat_neg.reset_index(inplace=True, drop=True)
    display(df_feat_neg)