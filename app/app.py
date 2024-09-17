from flask import Flask, render_template, session, redirect, url_for
from src.mvp_predictor.mvp_pred import mvp_predictions
from datetime import datetime
import pandas as pd
import os
import requests

app = Flask(__name__)
app.secret_key = os.urandom(24)

def get_player_faces(year):
    nba_data_url = f'http://data.nba.net/data/10s/prod/v1/{year}/players.json'
    response = requests.get(nba_data_url)

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

@app.route('/')
def home():
    return render_template('index.html')

# Default to latest year 
@app.route('/mvp_predictor/')
def mvp_predictor():
    cur_year = datetime.now().year
    # Convert DataFrame to HTML
    df = mvp_predictions(cur_year, cur_year)
    columns = df.columns.tolist()
    data = df.values.tolist()

    return render_template('mvp_predictor.html', selected_year=cur_year-1, columns=columns, data=data)

@app.route('/mvp_predictor/<int:year>')
def mvp_predictor_year(year):
    session['selected_year'] = year
    # Convert DataFrame to HTML
    df = mvp_predictions(year, year)
    columns = df.columns.tolist()
    data = df.values.tolist()
    players = get_player_faces(year)

    return render_template('mvp_predictor.html', selected_year=year-1, columns=columns, data=data, players=players)

@app.route('/game_predictor')
def game_predictor():
    return "<h1>Game Predictor Page</h1>"
        
if __name__ == '__main__':
    app.run(debug=True)
