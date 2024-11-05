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
        '2024-25': '',
        '2023-24': 'Nikola Jokić',
        '2022-23': 'JOEL EMBIID',
        '2021-22': 'Nikola Jokić',
        '2020-21': 'Nikola Jokić',
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

def train_mvp_model(start_year, end_year):
    if start_year < 1997 or start_year > 2025:
        print('Invalid season, there are no stats available for seasons:', start_year)
        return
    if end_year < 1997 or end_year > 2025:
        print('Invalid season, there are no stats available for seasons:', end_year)
        return

    seasons = [f"{year-1}-{str(year)[-2:]}" for year in range(start_year, end_year + 1)]
    mvp_winners = get_mvp_winners()

    training_data = pd.DataFrame()

    for season in seasons:
        # Load season data
        df_basic = get_player_stats(season)
        df_advanced = get_player_advanced_stats(season)

        # Remove unnecessary columns
        cols_to_remove = ['WNBA_FANTASY_PTS', 'NICKNAME'] + [col for col in df_basic.columns if 'RANK' in col]
        df_basic = df_basic.drop(columns=cols_to_remove, errors='ignore')

        # Cherry pick useful columns
        selected_cols_df_advanced = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'TS_PCT', 'USG_PCT', 'PIE']
        df_advanced = df_advanced[selected_cols_df_advanced]

        # Merge basic and advanced stats on common keys
        df_season = pd.merge(df_basic, df_advanced, on=['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN'], how='inner')
        df_season.insert(5, 'SEASON', season)

        # Label MVP winners for training
        df_season['MVP'] = df_season['PLAYER_NAME'].apply(lambda name: 1 if name.upper() == mvp_winners.get(season, "").upper() else 0)  

        # Feature engineering: add "counting stats" feature for quick MVP estimation
        df_season['COUNTING_STATS'] = df_season['PTS'] + df_season['REB'] + df_season['AST']
        df_season = df_season[df_season['COUNTING_STATS'] >= 30]  # Filter non-candidates

        # Collect all data
        training_data = pd.concat([training_data, df_season], ignore_index=True)

    X_train, y_train = df_season.drop(columns=['MVP', 'PLAYER_NAME', 'PLAYER_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON'], errors='ignore'), df_season['MVP']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler

def predict_season_mvp(model, scaler, year):
    season = f"{year-1}-{str(year)[-2:]}"
    
    # Load season data
    df_basic = get_player_stats(season)
    df_advanced = get_player_advanced_stats(season)

    # Remove unnecessary columns
    cols_to_remove = ['WNBA_FANTASY_PTS', 'NICKNAME'] + [col for col in df_basic.columns if 'RANK' in col]
    df_basic = df_basic.drop(columns=cols_to_remove, errors='ignore')

    selected_cols_df_advanced = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'TS_PCT', 'USG_PCT', 'PIE']
    df_advanced = df_advanced[selected_cols_df_advanced]

    # Merge basic and advanced stats on common keys
    df_current = pd.merge(df_basic, df_advanced, on=['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN'], how='inner')

    # Feature engineering: add "counting stats" feature for quick MVP estimation
    df_current['COUNTING_STATS'] = df_current['PTS'] + df_current['REB'] + df_current['AST']
    df_current = df_current[df_current['COUNTING_STATS'] >= 30]  # Filter non-candidates

    # Prepare features
    X_current = df_current.drop(columns=['PLAYER_NAME', 'PLAYER_ID', 'TEAM_ID', 'TEAM_ABBREVIATION'])
    X_current_scaled = scaler.transform(X_current)

    # Predict from model
    predictions = model.predict(X_current_scaled)

    # Normalize the probabilities and add to DataFrame
    df_current['PROB(%)'] = (predictions / predictions.sum()) * 100
    df_current['PROB(%)'] = df_current['PROB(%)'].round(3)
    
    # Sort players by highest MVP PROB
    df_current.sort_values(by='PROB(%)', ascending=False, inplace=True)

    # Extra work to make the dataframe more pleasing to look at on website
    df_current = df_current.rename(columns={'TEAM_ABBREVIATION': 'TEAM', 'W_PCT': 'W(%)', 'FG_PCT': 'FG(%)', 'FG3_PCT': 'FG3(%)', 'FT_PCT': 'FT(%)', 'TS_PCT': 'TS(%)', 'USG_PCT': 'USG(%)',
                                            'PLUS_MINUS': '+/-', 'PLAYER_NAME': 'PLAYER', 'OFF_RATING': 'OFF_RTG', 'DEF_RATING': 'DEF_RTG', 'NET_RATING': 'NET_RTG'})
    
    #df_current['MVP'] = df_current['PLAYER'].apply(lambda name: 1 if name.upper() == mvp_winners.get(season, "").upper() else 0)
    df_current = df_current.drop(columns={'PLAYER_ID', 'TEAM_ID', 'NBA_FANTASY_PTS', 'COUNTING_STATS', 'BLKA', 'PF', 'PFD'})
    df_current.insert(0, 'RANK', range(1, len(df_current.index)+1))

    return df_current.head(10) # Return the top 10 candidates
