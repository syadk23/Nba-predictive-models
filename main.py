from sklearn.metrics import accuracy_score
from src.nba_game_predictor.game_pred import game_prediction
from src.mvp_predictor.mvp_pred import mvp_predictions
import pandas as pd
  
def main():
    #pd.set_option('display.max_columns', None)

    mvp_predictions(start_year=2000, end_year=2009)
    #print(df)
        
    """ game_pred_model_predictions = game_prediction()
    acc = accuracy_score(game_pred_model_predictions['actual'], game_pred_model_predictions['predictions'])
    print(acc) """

if __name__ == '__main__':
    main()