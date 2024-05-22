import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler

def add_target(team):
    team['target'] = team['won'].shift(-1)
    return team

def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    seasons = sorted(data['season'].unique())
    for i in range(start, len(seasons), step):
        season = seasons[i]

        train = data[data['season'] < season]
        test = data[data['season'] == season]

        model.fit(train[predictors], train['target'])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)

        combined = pd.concat([test['target'], preds], axis=1)
        combined.columns = ['actual', 'predictions']
        
        all_predictions.append(combined)
    
    return pd.concat(all_predictions)

def find_team_averages(team):
    rolling = team.loc[:, team.columns != 'team'].rolling(10).mean()
    return rolling

def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby('team', group_keys=False).apply(lambda x: shift_col(x, col_name))

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