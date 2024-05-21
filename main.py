from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.nba_game_predictor.game_pred import *
from src.mvp_predictor.mvp_pred import *
import pandas as pd
import numpy as np
from collections import defaultdict

def game_prediction():
    # data processing
    df = pd.read_csv('data/nba_games.csv')
    df = df.groupby('team', group_keys=False).apply(add_target)
    df['target'][pd.isnull(df['target'])] = 2
    df['target'] = df['target'].astype(int, errors='ignore')

    nulls = pd.isnull(df)
    nulls = nulls.sum()
    nulls = nulls[nulls > 0]

    valid_cols = df.columns[~df.columns.isin(nulls.index)]
    df = df[valid_cols].copy()

    rr = RidgeClassifier(alpha=1)
    split = TimeSeriesSplit(n_splits=3)
    sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction='forward', cv=split)
    scaler = MinMaxScaler()

    removed_cols =  ['season', 'date', 'won', 'target', 'team', 'team_opp']
    selected_cols = df.columns[~df.columns.isin(removed_cols)]
    df[selected_cols] = scaler.fit_transform(df[selected_cols])

    # Early model without rolling data and without home court stats ~57% accuracy
    """ sfs.fit(df[selected_cols], df['target'])
    predictors = list(selected_cols[sfs.get_support()])
    predictions = backtest(df, rr, predictors)
    predictions = predictions[predictions['actual'] != 2] """
    
    #print(accuracy_score(predictions['actual'], predictions['predictions']))
    #print(df.groupby('home').apply(lambda x: x[x['won'] == 1].shape[0] / x.shape[0]))

    df_rolling = df[list(selected_cols) + ['won', 'team', 'season']]
    df_rolling = df_rolling.groupby(['team', 'season'], group_keys=False).apply(find_team_averages)

    rolling_cols = [f'{col}_rolling10' for col in df_rolling.columns]
    df_rolling.columns = rolling_cols
    
    df = pd.concat([df, df_rolling], axis=1)
    df = df.dropna()

    df['home_next'] = add_col(df, 'home')
    df['team_opp_next'] = add_col(df, 'team_opp')
    df['date_next'] = add_col(df, 'date')
    df = df.copy()

    df_full = df.merge(df[rolling_cols + ['team_opp_next', 'date_next', 'team']], left_on=['team', 'date_next'], right_on=['team_opp_next', 'date_next'])

    removed_cols = list(df_full.columns[df_full.dtypes == 'object']) + removed_cols
    selected_cols = df_full.columns[~df_full.columns.isin(removed_cols)]
   
    # machine learning
    sfs.fit(df_full[selected_cols], df_full['target'])
    predictors = list(selected_cols[sfs.get_support()])
    predictions = backtest(df_full, rr, predictors)

    #print(df_full.groupby('home').apply(lambda x: x[x['won'] == 1].shape[0] / x.shape[0])) # Final model ~ 57% accurate 

    return predictions
  
     
def main():
    start_year = 2018
    end_year = 2023
    seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year + 1)]

    mvp_winners = get_mvp_winners()
    df_full = pd.DataFrame()

    #pd.set_option('display.max_columns', None)

    for season in seasons:
        df_basic = get_player_stats(season)
        df_advanced_stats = get_player_advanced_stats(season)

        df_basic.insert(len(df_basic.columns), 'COUNTING_STATS', df_basic['PTS'] + df_basic['REB'] + df_basic['AST'])

        removed_cols = []
        remove_string = 'RANK'
        for col in df_basic.columns:
            if remove_string in col:
                removed_cols.append(col)
        removed_cols.append('WNBA_FANTASY_PTS')
        removed_cols.append('NICKNAME')

        selected_cols = df_basic.columns[~df_basic.columns.isin(removed_cols)]
        df_basic = df_basic[selected_cols]

        selected_cols_df_advanced_stats = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'TS_PCT', 'USG_PCT', 'PIE']
        df_advanced_stats = df_advanced_stats[selected_cols_df_advanced_stats]
        
        # IF COUNTING STATS ARE < 30 or games played is < 30, DISREGARD PLAYER AS THERE HAS NEVER BEEN A CASE FOR THEM TO WIN MVP, ALSO HELPS LOWER THE AMOUNT OF DATA BEING USED
        df_basic = df_basic.drop(df_basic[df_basic['COUNTING_STATS'] < 30.0].index)
        df_basic = df_basic.drop(df_basic[df_basic['GP'] < 30].index)

        df_season = pd.merge(df_basic, df_advanced_stats, how='inner')
        df_season.insert(5, 'SEASON', season)
        df_season.insert(len(df_season.columns), 'WON_MVP', 0)
        df_season['WON_MVP'] = df_season['PLAYER_NAME'].apply(lambda str: str.upper() in mvp_winners[season]).map({True: 1, False: 0})
  


    #game_pred_model_predictions = game_prediction()
    #acc = accuracy_score(game_pred_model_predictions['actual'], game_pred_model_predictions['predictions'])
    #print(acc)

if __name__ == '__main__':
    main()