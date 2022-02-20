# A series of percentage calculations
# Author: Nathan Fragar

def calculate_percentage(df, numerator_col, denominator_col, result_col, round_result=1):
    """Clean data and Calculate Percentage of Numerator divided by Denominator
    
        1. Check Numerator is not greater than Denominator, If so, clear and populate mean
        2. Calculate percentage of  Numerator divided by Denominator rounded to the number of decimal places
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    numerator_col: str
        Numerator Column Name
    denominator_col: str
        Denominator Column Name
    result_col: str
        Result Column Name
    round_result: integer
        Rounding to a decimal

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing all features with updated column
    """
    import pandas as pd
    import numpy as np
    
    df[numerator_col] = df[numerator_col].astype(float)
    df[denominator_col] = df[denominator_col].astype(float)
    
    
    df.loc[df[denominator_col] < df[numerator_col],'Invalid'] = True
    
    df.loc[df['Invalid'] == True,numerator_col] = df[numerator_col].mean()
    df.loc[df['Invalid'] == True,denominator_col] = df[denominator_col].mean()
    
    df.loc[df['Invalid'] == True,numerator_col] = df[numerator_col].round(round_result)
    df.loc[df['Invalid'] == True,denominator_col] = df[denominator_col].round(round_result)
    
    # Calculate percentage
    df.loc[df[denominator_col] == 0,result_col] = 0
    
    df.loc[df[numerator_col] < df[denominator_col],result_col] = (df[numerator_col] / df[denominator_col]) * 100
    df[result_col] = df[result_col].round(round_result)
    
    df.drop('Invalid', axis=1, inplace=True)
                  
    return df