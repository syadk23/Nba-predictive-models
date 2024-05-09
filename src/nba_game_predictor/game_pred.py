import pandas as pd

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
