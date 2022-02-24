# A series of percentage calculations
# Author: Nathan Fragar

def calculate_feature(df, left_col, right_col, result_col, operator, round_result=1):
    """ Calculate feature result
    
        1. Calculate result_col = left_col operator right_col 
        2. Round Result
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    left_col: str
        Column left of operator
    right_col: str
        Column Right of operator
    result_col: str
        Result Column Name
    operator: str
        Supports Addition (+), Subtraction (-), Multiplication
    round_result: integer
        Rounding to a decimal

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing all features with updated column
    """
    import pandas as pd
    import numpy as np
    
    # Perform Calculation
    # Addition
    if    (operator == '+'):
        df[result_col] = df[left_col] + df[right_col]
    # Subtraction
    elif (operator == '-'):
        df[result_col] = df[left_col] - df[right_col]
    # Multiplication
    elif (operator == '*'):
        df[result_col] = df[left_col] * df[right_col]

    df[result_col] = df[result_col].round(round_result)
    
    return df