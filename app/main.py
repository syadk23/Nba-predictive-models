from sklearn.metrics import accuracy_score
from src.nba_game_predictor.game_predictor import game_prediction
from src.mvp_predictor.mvp_predictor import *
import pandas as pd
  
def main():
    #pd.set_option('display.max_columns', None)

    model, scaler = train_mvp_model(start_year=2018, end_year=2022)
    df = predict_season_mvp(model, scaler, 2023)
    print(df)        
    """ game_pred_model_predictions = game_prediction()
    acc = accuracy_score(game_pred_model_predictions['actual'], game_pred_model_predictions['predictions'])
    print(acc) """

if __name__ == '__main__':
    main()