import pandas as pd
import numpy as np

# Create anomalous flag columns and replace the anomalous values with nan
# if col 0 >col 0, fill 0, 1, 2 with np.nan and create anom columns.
def add_anom_higher (df_in, cal_variables):
    for col in cal_variables:
        if (df_in[col[0]] > df_in[col[1]]).any():
            for i in range(3):
                df_in[f'{col[i]}_anom'] = df_in[col[0]] > df_in[col[1]]
                df_in[f'{col[i]}_anom'] = df_in[f'{col[i]}_anom'].replace({True: 1, False: 0})
            df_in.loc[df_in[col[0]] > df_in[col[1]], [col[0], col[1], col[2]]] = np.nan 
    


def add_anom_negative (df_in, variables): 
    for col in variables:
        if (df_in[col] < 0).any():
            if f'{col}_anom' in df_in.columns:
                df_in.loc[df_in[col] < 0, f'{col}_anom'] = 1
            else:
                df_in[f'{col}_anom'] = df_in[col] < 0
                df_in[f'{col}_anom'] = df_in[f'{col}_anom'].replace({True: 1, False: 0})
            df_in.loc[df_in[col] < 0, col] = np.nan
    
    
def add_anom_positive (df_in, max_dic): 
    for col in max_dic:
        if (df_in[col] > max_dic[col]).any():
            if f'{col}_anom' in df_in.columns:
                df_in.loc[df_in[col] > max_dic[col], f'{col}_anom'] = 1
            else:      
                df_in[f'{col}_anom'] = df_in[col] > max_dic[col]
                df_in[f'{col}_anom'] = df_in[f'{col}_anom'].replace({True: 1, False: 0})
            df_in.loc[df_in[col] > max_dic[col], col] = np.nan    

        
def add_logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)
        res.columns.values[m] = l + '_log'
        m += 1
    return res


def add_squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l] * res[l]).values)
        res.columns.values[m] = l + '_sq'
        m += 1
    return res


def add_knowledge(df):

    # Free-Throw Rate: FTr = FTA / FGA
    # Free throw percentage is an indicator of offensive efficiency. Free throws are one of the most effective ways to score, along with 3-pointers and close-to-the-basket shooting. The more free throws you can get per shot, the more effective the offense will be.
    df['FTr'] = df['FTA'] / df['FGA']

    # Free Throw Made Rate: FTMr = FTM / FGM
    df['FTMr'] = df['FTM'] / df['FGM']

    # 3_Point Attempt Rate: 3PAr = 3PA / FGA
    df['3PAr'] = df['3PA'] / df['FGA']

    # 3_Point Made Rate: 3PMr = 3P Made / FGM
    df['3PMr'] = df['3P Made'] / df['FGM']

    # Assist to Turnover Ratio: AST/TOV = AST / TOV
    df['AST/TOV'] = df['AST'] / df['TOV']

    # Assist Ratio: ASTR = (AST ÷ (FGA + (0.44 x FTA) + AST + TO)) x 100%
    df['ASTR'] = (df['AST'] /
                            (df['FGA'] + (0.44 * df['FTA']) +
                             df['AST'] + df['TOV'])) * 100

    # Average point per minutes: APM = PTS / MIN
    df['APM'] = df['PTS'] / df['MIN']

    # FT_miss = FTA - FTM
    df['FT_miss'] = df['FTA'] - df['FTM']

    # FG_miss = FGA - FGM
    df['FG_miss'] = df['FGA'] - df['FGM']

    # 3P_miss =  3PA - 3P Made
    df['3P_miss'] = df['3PA'] - df['3P Made']

    # Defensive Rebound Percentage: DRB% = (Player Defensive Rebounds x (Team Minutes Played ÷ 5)) ÷ (Player Minutes Played x (Team Defensive Rebounds + Opponent Offensive Rebounds)) * 100
    # Missing features: assume Team Minutes Played = 240, Team Defensive Rebounds = Opponent Offensive Rebounds = 30
    df['DRB%'] = (df['DREB'] *
                            (240 / 5)) / (df['MIN'] * (30 + 30)) * 100

    # Offensive Rebound Percentage: ORB% = ((Player Offensive Rebounds x (Team Minutes Played ÷ 5)) ÷ (Player Minutes Played x (Team Offensive Rebounds + Opponent Defensive Rebounds))) * 100
    # Missing features: assume Fouls Drawn = 0, Team Minutes Played = 240, Team Offensive Rebounds = Opponent Defensive Rebounds = 30 ？？？
    df['ORB%'] = (df['OREB'] *
                            (240 / 5)) / (df['MIN'] * (30 + 30)) * 100

    # Efficiency (EFF) = (PTS + REB + AST + STL + BLK – Missed FG – Missed FT – TO)
    df['EFF'] = df['PTS'] + df['REB'] + df[
        'AST'] + df['STL'] + df['BLK'] - df[
            'FG_miss'] - df['FT_miss'] - df['TOV']

    # Efficiency per Game: EFF/GP
    df['EFF/GP'] = df['EFF'] / df['GP']

    # Efficiency per 48 minutes: (EFF/48) = (EFF ÷ Minutes Played) x 48
    df['EFF/48'] = (df['EFF'] / df['MIN']) * 48

    # Efficiency per 40 minutes: (EFF/40) = (EFF ÷ Minutes Played) x 40
    df['EFF/40'] = (df['EFF'] / df['MIN']) * 40

    # Effective Field Goal Percentage: eFG% = (FGA + 0.5 * 3PA) / FGA
    # Effective field goal percentage measures shooting efficiency strictly in terms of regular shots other than free throws.
    # An adjuste field goal percentage that takes into account the extra point provided by the 3-point shot.
    df['eFG%'] = (df['FGA'] +
                            0.5 * df['3PA']) / df['FGA']

    # Game Score = Points Scored + (0.4 x Field Goals) – (0.7 x Field Goal Attempts) – (0.4 x (Free Throw Attempts – Free Throws)) + (0.7 x Offensive Rebounds) + (0.3 x Defensive Rebounds) + Steals + (0.7 x Assists) + (0.7 x Blocks) – (0.4 x Personal Fouls) – Turnovers
    # Missing features: assume Personal Fouls = 2
    df['GS'] = df['PTS'] + (0.4 * df['FGM']) - (
        0.7 * df['FGA']
    ) - (0.4 * (df['FTA'] - df['FTM'])) + (
        0.7 * df['OREB']) + (
            0.3 * df['DREB']) + df['STL'] + (
                0.7 * df['AST']) + (0.7 * df['BLK']) - (
                    0.4 * 2) - df['TOV']

    # Per 36 Minutes:
    df['per_36'] = df['AST'] / df['MIN'] * 36

    # Performance Index Ratin: PIR = (Points + Rebounds + Assists + Steals + Blocks + Fouls Drawn) – (Missed Field Goals + Missed Free Throws + Turnovers + Shots Rejected + Fouls Committed)
    # Missing features: assume Fouls Drawn = 0, Shots Rejected = 0, Fouls Committed = 0
    df['PIR'] = (df['PTS'] + df['REB'] +
                           df['AST'] + df['STL'] +
                           df['BLK'] + 0) - (df['FG_miss'] +
                                                       df['FT_miss'] +
                                                       df['TOV'] + 0 + 0)

    # Player Efficiency Rating (PER)
    # from John Hollinger , it's a catch_all feature. PER measures a player's productivity per minute. Its scoring system (statistical point value system) adds up all the positive contributions of a player to the team and subtracts all the negative ones. Its calculation can be adjusted to the pace of the game and the time spent on the field (e.g. converted to per 100 rounds or per minute data) to make it easier to compare players.
    # Its drawback is that there is little reliable defensive data that can be entered into the formula. We all know that steals and blocks do not necessarily correspond to good defense. So in the PER system, defensive specialists are at a disadvantage, and good offensive and defensive players may be ranked lower than players who only play offense.
    # No data: assume missing foul
    df['PER'] = df['FGM'] * 85.910 + df[
        'STL'] * 53.897 + df['3P Made'] * 51.757 + df[
            'FTM'] * 46.845 + df['BLK'] * 39.190 + df[
                'OREB'] * 39.190 + df['AST'] * 34.677 + df[
                    'DREB'] * 14.707 - 2 * 17.174 - df[
                        'FT_miss'] * 20.091 - df[
                            'FG_miss'] * 39.190 - df['TOV'] * 53.897 * (
                                1 / df['MIN'])

    # postitive player efficiency rating
    # # Missing features: assume Foul = 2
    df['PPER'] = df['FGM'] * 85.910 + df[
        'STL'] * 53.897 + df['3P Made'] * 51.757 + df[
            'FTM'] * 46.845 + df['BLK'] * 39.190 + df[
                'OREB'] * 39.190 + df['AST'] * 34.677 + df[
                    'DREB'] * 14.707 * (1 / df['MIN'])

    # negative player efficiency rating
    df['NPER'] = 2 * 17.174 + df[
        'FT_miss'] * 20.091 + df['FG_miss'] * 39.190 + df[
            'TOV'] * 53.897 * (1 / df['MIN'])

    # Points per Possession: PPP = Points ÷ (Field Goal Attempts + (0.44 x Free Throw Attempts) + Turnovers)
    df['PPP'] = df['PTS'] / (df['FGA'] +
                                                 (0.44 * df['FTA']) +
                                                 df['TOV'])

    # Rebound Rate: (100 x Rebounds x Team Minutes Played) ÷ (Player Minutes Played x (Team Total Rebounds + Opposing Team Total Rebounds))
    # Missing features: assume Team Minutes = 48, Team Total Rebounds = Opposing Team Total Rebounds = 30
    df['REBr'] = (100 * df['REB'] *
                            48) / (df['MIN'] * (30 + 30))

    # Steal Percentage: STL% = (100 x Player Steals x Team Minutes) ÷ (Player Minutes Played x Opponent Team Possessions)
    # Missing features: assume Team Minutes = 48, Opponent Team Possessions = 30
    df['STL%'] = (100 * df['STL'] *
                            48) / (df['MIN'] * 30)

    # Steal/Turnover Ratio: STL/TOV = Steals ÷ Turnovers
    df['STL/TOV'] = df['STL'] / df['TOV']

    # True Shooting Percentage: TS% = Points ÷ ((2 x Field Goal Attempts) + (0.88 x Free Throw Attempts)
    # A measure of shooting efficiency that includes 2-pointers, 3-pointers, and free throws.
    df['TS%'] = df['PTS'] / (2 * (df['FGA']) + 0.88 * (df['FTA']))

    # Turnover Percentage: TOV% = (TOV ÷ (FGA + (0.44 x FTA) + TOV)) x 100%
    # available since the 1977-78 season in the NBA
    df['TOV%'] = (
        df['TOV'] /
        (df['FGA'] +
         (0.44 * df['FTA']) + df['TOV'])) * 100

    # Usage Rate = ( (Field Goal Attempts + (0.44 x Free Throw Attempts) + (o.33 x Assists) + Turnovers) x 40 x League Pace ) ÷ (Minutes Played x Team Pace)
    # Missing features: assume League Pace = 100 and Team Pace = 100
    df['USGr'] = ((df['FGA'] + (0.44 * df['FTA']) +
                             (0.33 * df['AST']) * df['TOV']) *
                            40 * 100) / (df['MIN'] * 100)

    # Value Added” (VA) VA = (Minutes Played x (Player Efficiency Rating – Position Replacement Level) ) ÷ 67
    # “Minutes Played” is the number of minutes of play the player has recorded.
    # “PER” is “Player Efficiency Rating” – an overall rating of a player’s per-minute statistical production.
    # “PRL” is “Position Replacement Level” – what a replacement player (for example the 12th ranking player on the roster) would produce. Power Forwards are 11.5, Point Guards are 11.0, Centers are 10.6, Shooting Guards and Small Forwards are 10.5.
    #  MIn in formular is total minutes. assume Position Replacement Level = 11
    df['VA'] = (df['MIN'] * df['GP'] * (df['PER'] - 11)) / 67

    # Estimated Wins Added” (EWA)
    df['EWA'] = df['VA'] / 30

    # Double-Double: DD two of the following categories: points, rebounds, assists, steals, and blocks.
    # Triple_Doubles: TD three of the following categories: points, rebounds, assists, steals, and blocks.
    # Quadruple-Doubles： QD four of the following categories: points, rebounds, assists, steals, and blocks.
    # Create boolean columns for each category in doubless_lists.
    # Then add these columns together to form a doubles list. (2 for Double_Double; 3 for Triple_Doubles etc.)

    doubles_list = ['PTS', 'REB', 'AST', 'STL', 'BLK']

    for feature in doubles_list:
        if (df[feature] > 10).any():
            df[f'{feature}_double'] = df[feature] >= 10
            df[f'{feature}_double'] = df[f'{feature}_double'].replace({True: 1, False: 0})

    doubles_feature = []
    for i in doubles_list:
        doubles_feature.append(i+'_double')

    df['double'] = 0
    for i in doubles_feature:
        try:
            df['double'] += df[i]
        except:
            pass
        
    return df