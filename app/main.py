from sklearn.metrics import accuracy_score
from src.nba_game_predictor.game_predictor import game_prediction
from src.mvp_predictor.mvp_predictor import *
import pandas as pd
import requests

def get_player_faces(year):
    nba_data_url = f'http://data.nba.net/data/10s/prod/v1/%7B%7Byear%7D%7D/players.json'
    response = requests.get(nba_data_url)
    print(response)

    if response.status_code == 200:
        players = response.json().get('league', {}).get('standard', [])
        
        # Extract relevant player data (e.g., name, and profile picture URL)
        player_images = []
        for player in players:
            if player.get('isActive'):
                player_id = player['personId']
                full_name = f"{player['firstName']} {player['lastName']}"
                # NBA typically has a pattern for player image URLs
                img_url = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"
                player_images.append({
                                    'id': player_id,
                                    'name': full_name,
                                    'img_url': img_url
                                })            
        return player_images
    return []
  
def main():
    #pd.set_option('display.max_columns', None)

    model, scaler = train_mvp_model(start_year=2022, end_year=2022)
    df = predict_season_mvp(model, scaler, 2023)
    print(get_player_faces(2023))

    """ game_pred_model_predictions = game_prediction()
    acc = accuracy_score(game_pred_model_predictions['actual'], game_pred_model_predictions['predictions'])
    print(acc) """

if __name__ == '__main__':
    main()