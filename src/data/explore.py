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


def print_duplicate(df_in, i):
    print('There are %d of duplicated columns.' % (len(df_in[i]) - len(df_in[i].unique())))
    df = df_in[df_in.duplicated(keep=False)]
    display(df)
    
    
def print_head(df_in):
    """Print head of data set

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe 

    Returns
    -------
    head of dataset, shape of dataset
    """
    df = df_in.head()
    display(df)
    print('Dataset shape: ', df_in.shape)


# Function to calculate missing values by column 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns.head(10)
