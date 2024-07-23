from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score
from nba_api.stats.endpoints import leaguedashplayerstats
from collections import defaultdict
import pandas as pd
import xgboost as xgb
import numpy as np

def get_player_stats(season):
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed='PerGame')
    stats = player_stats.get_data_frames()[0]
    return stats

def get_player_advanced_stats(season):
    player_advanced_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame')
    advanced_stats = player_advanced_stats.get_data_frames()[0]
    return advanced_stats

def get_player_avg_stats(stats, player_name=None):
    player_avg_stats = stats[['PLAYER_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'TOV', 'PLUS_MINUS']]
    if player_name:
        player_avg_stats = player_avg_stats[player_avg_stats['PLAYER_NAME'].str.contains(player_name, case=False)]
    return player_avg_stats

def get_mvp_winners():
    mvp_winners = {
        '2023-24': 'NIKOLA JOKIC',
        '2022-23': 'JOEL EMBIID',
        '2021-22': 'NIKOLA JOKIC',
        '2020-21': 'NIKOLA JOKIC',
        '2019-20': 'GIANNIS ANTETOKOUNMPO',
        '2018-19': 'GIANNIS ANTETOKOUNMPO',
        '2017-18': 'JAMES HARDEN',
        '2016-17': 'RUSSELL WESTBROOK',
        '2015-16': 'STEPHEN CURRY',
        '2014-15': 'STEPHEN CURRY',
        '2013-14': 'KEVIN DURANT',
        '2012-13': 'LEBRON JAMES',
        '2011-12': 'LEBRON JAMES',
        '2010-11': 'DERRICK ROSE',
        '2009-10': 'LEBRON JAMES',
        '2008-09': 'LEBRON JAMES',
        '2007-08': 'KOBE BRYANT',
        '2006-07': 'DIRK NOWITZKI',
        '2005-06': 'STEVE NASH',
        '2004-05': 'STEVE NASH',
        '2003-04': 'KEVIN GARNETT',
        '2002-03': 'TIM DUNCAN',
        '2001-02': 'TIM DUNCAN',
        '2000-01': 'ALLEN IVERSON',
        '1999-00': 'SHAQUILLE O\'NEAL',
        '1998-99': 'KARL MALONE',
        '1997-98': 'MICHAEL JORDAN',
        '1996-97': 'KARL MALONE',
        '1995-96': 'MICHAEL JORDAN',
        '1994-95': 'DAVID ROBINSON',
        '1993-94': 'HAKEEM OLAJUWON',
        '1992-93': 'CHARLES BARKLEY',
        '1991-92': 'MICHAEL JORDAN',
        '1990-91': 'MICHAEL JORDAN',
        '1989-90': 'MAGIC JOHNSON',
        '1988-89': 'MAGIC JOHNSON',
        '1987-88': 'MICHAEL JORDAN',
        '1986-87': 'MAGIC JOHNSON'
    }
    return mvp_winners

def mvp_predictions(start_year, end_year):
    if start_year < 1987 or start_year > 2024:
        print('Invalid season, there are no stats available for seasons:', start_year)
        return
    if end_year < 1987 or end_year > 2024:
        print('Invalid season, there are no stats available for seasons:', end_year)
        return

    seasons = [f"{year-1}-{str(year)[-2:]}" for year in range(start_year, end_year + 1)]

    mvp_winners = get_mvp_winners()

    for season in seasons:
        df_basic = get_player_stats(season)
        df_advanced_stats = get_player_advanced_stats(season)

        df_basic.insert(len(df_basic.columns), 'COUNTING_STATS', df_basic['PTS'] + df_basic['REB'] + df_basic['AST'])
        df_basic['COUNTING_STATS'] = df_basic['COUNTING_STATS'].apply(lambda x: round(x, 1))
        df_basic['AGE'] = df_basic['AGE'].astype(int)

        removed_cols = ['WNBA_FANTASY_PTS', 'NICKNAME']
        for col in df_basic.columns:
            if 'RANK' in col:
                removed_cols.append(col)

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
  
        X, y = df_season.drop(columns=['WON_MVP', 'PLAYER_NAME', 'PLAYER_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON']), df_season['WON_MVP']

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        """ scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=6, random_state=42)
        model.fit(X_train_scaled, y_train)
 """
        regr = RandomForestRegressor(max_depth=3, random_state=42)
        regr.fit(X, y)

        """ y_predict = model.predict(X_test_scaled)
        y_predict_probs = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_predict)

        cur_season_scaled = scaler.transform(df_season.drop(columns=['WON_MVP', 'PLAYER_NAME', 'PLAYER_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON']))
        cur_season_predictions = model.predict_proba(cur_season_scaled)[:, 1] """

        regr_seasons_predictions = regr.predict(X)

        df_season['PREDICTED_PROBABILITY'] = regr_seasons_predictions * 100
        prob_sum = df_season['PREDICTED_PROBABILITY'].sum()

        df_season['PREDICTED_PROBABILITY'] = df_season['PREDICTED_PROBABILITY'] / prob_sum * 100 if prob_sum > 0 else 0
        df_season['PREDICTED_PROBABILITY'] = df_season['PREDICTED_PROBABILITY'].apply(lambda x: round(x, 3))
        df_season.sort_values(by='PREDICTED_PROBABILITY', ascending=False, inplace=True)

        # Extra work to make the dataframe more pleasing to look at on website
        hide_cols = ['PLAYER_ID', 'TEAM_ID']
        visible_cols = df_season.columns[~df_season.columns.isin(hide_cols)]
        df_season = df_season[visible_cols]
        df_season = df_season.rename(columns={'TEAM_ABBREVIATION': 'TEAM'})

        return df_season




        #feat_imp = regr.feature_importances_
        #print(feat_imp)

        #print(df_season)
        #print(df_season[['PLAYER_NAME', 'SEASON', 'PREDICTED_PROBABILITY', 'WON_MVP']])
