from nba_api.stats.endpoints import leaguedashplayerstats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pprint

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
    # Use last 10 as testing for now
    mvp_winners = {
        '2022-23': 'Joel Embiid',
        '2021-22': 'Nikola Jokic',
        '2020-21': 'Nikola Jokic',
        '2019-20': 'Giannis Antetokounmpo',
        '2018-19': 'Giannis Antetokounmpo',
        '2017-18': 'James Harden',
        '2016-17': 'Russell Westbrook',
        '2015-16': 'Stephen Curry',
        '2014-15': 'Stephen Curry',
        '2013-14': 'Kevin Durant',
        '2012-13': 'LeBron James'
    }
    return mvp_winners

def add_target(team):
    team['target'] = team['won'].shift(-1)
    return team

def main():
    start_year = 2012
    end_year = 2023
    seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year + 1)]

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

    # machine learning
    rr = RidgeClassifier(alpha=1)
    split = TimeSeriesSplit(n_splits=3)
    sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction='forward', cv=split)
    scaler = MinMaxScaler()

    removed_cols =  ['season', 'date', 'won', 'target', 'team', 'team_opp']
    selected_cols = df.columns[~df.columns.isin(removed_cols)]
    df[selected_cols] = scaler.fit_transform(df[selected_cols])

    sfs.fit(df[selected_cols], df['target'])
    predictors = list(selected_cols[sfs.get_support()])
    
    if __name__ == "__main__":
        main()
