# NBA Calculation Ratings
# Author: Nathan Fragar
# Algorithm is an interpretation of Hollinger's PER algorithm by Zach Fein - Last Viewed 2022-02-20
# https://bleacherreport.com/articles/113144-cracking-the-code-how-to-calculate-hollingers-per-without-all-the-mess

def calculate_player_efficiency_rating(df):
    """Calculate an add Player Efficiency Ratings
    
        PPER = Positive Player Efficiency Ratings
        NPER = Negative Player Efficiency Ratings
        PER  = Player Efficiency Ratings = PPER - NPER
        
        
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing all features with updated column
    """
    import pandas as pd
    import numpy as np

    # Variables used to calculate PER
    rounding_integer = 1
    # Offensive Ratios
    FTM_ratio  = 46.845
    FGM_ratio  = 85.910
    ThreePTM_ratio = 51.757
    AST_ratio = 34.677
    STL_ratio = 53.897
    BLK_ratio       = 39.190
    OREB_ratio      = 39.190
    DREB_ratio      = 14.707    
    
    # Offensive Ratios
    FT_Missed_ratio = 20.091
    FG_Missed_ratio = 39.190
    TO_ratio        = 53.897
    
    # Calculate Positive Player Efficiency Rating
    #df['PPER'] = (((df['FTM'] * FTM_ratio) + (df['FGM'] * FGM_ratio) + (df['3PTM'] * ThreePTM_ratio) + (df['AST'] * AST_ratio) + (df['STL'] * STL_ratio) + (df['BLK'] * BLK_ratio) + (df['OREB'] * OREB_ratio) + (df['DREB'] * DREB_ratio)) * (1/df['MIN']))

    df['PPER'] = (1/df['MIN']) * ((df['FTM'] * FTM_ratio) + (df['FGM'] * FGM_ratio ) + (df['3P Made'] * ThreePTM_ratio) + (df['AST'] * AST_ratio) + (df['STL'] * STL_ratio) + (df['BLK'] * BLK_ratio) + (df['OREB'] * OREB_ratio) + (df['DREB'] * DREB_ratio)  )
    
    df['PPER'] = df['PPER'].round(1)
    
    # Calculate Negative Player Efficiency Rating
    #df['NPER'] = (((df['FG_Missed'] * FT_Missed_ratio) + (df['FG_Missed'] * FG_Missed_ratio) + (df['TO'] * TO_ratio)) * (1/df['MIN']))
    df['NPER'] = (1/df['MIN']) * ((df['FG_Missed'] * FT_Missed_ratio) + (df['FG_Missed'] * FG_Missed_ratio) + (df['TOV'] * TO_ratio))
    
    df['NPER'] = df['NPER'].round(rounding_integer)
    
    # Calculate Player Efficiency Rating
    df['PER']     =  df['PPER'] - df['NPER']
    df['PER']     = df['PER'].round(rounding_integer)
    
    return df