# Author: Nathan Fragar

def null_numeric_values(df, cols, operator, target_value ):
    """Update Columns with Null value if value meets operator condition, and then apply the mean for that column
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : str
        Name of the target column
    operator : str
        Operator to determine if a column meets value condition
    target_value : float
        Value used to to meet condition

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing all features with updated column
    """
    import pandas as pd
    import numpy as np
    
    rounding_integer = 1
    
    for col in cols:
        # Apply condition to column values
        if col in df.columns:
            if operator == '<':
                df.loc[df[col] < target_value,col ] = np.nan
            elif operator == '<=':
                df.loc[df[col] <= target_value,col ] = np.nan
            elif operator == '==':
                df.loc[df[col] == target_value,col ] = np.nan
            elif operator == '>':
                df.loc[df[col] > target_value,col ] = np.nan
            elif operator == '>=':
                df.loc[df[col] >= target_value,col ] = np.nan     
            
        # Apply mean to null values
        df[col].fillna((df[col].mean()), inplace=True)
        
        df[col]= df[col].round(rounding_integer)
                   
    return df