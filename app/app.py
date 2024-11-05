from flask import Flask, render_template, session, redirect, url_for
from src.mvp_predictor.mvp_predictor import train_mvp_model, predict_season_mvp
from datetime import datetime
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

model, scaler = train_mvp_model(2018, 2022)

@app.route('/')
def home():
    return render_template('index.html')

# Default to latest year 
@app.route('/mvp_predictor/')
def mvp_predictor():
    cur_month = datetime.now().month
    cur_year = datetime.now().year
    if cur_month > 9:
        cur_year+=1

    # Convert DataFrame to HTML
    df = predict_season_mvp(model, scaler, cur_year)
    columns = df.columns.tolist()
    data = df.values.tolist()

    return render_template('mvp_predictor.html', selected_year=cur_year, columns=columns, data=data)

@app.route('/mvp_predictor/<int:year>')
def mvp_predictor_year(year):
    session['selected_year'] = year

    # Convert DataFrame to HTML
    df = predict_season_mvp(model, scaler, year)
    columns = df.columns.tolist()
    data = df.values.tolist()

    return render_template('mvp_predictor.html', selected_year=year, columns=columns, data=data)

@app.route('/game_predictor')
def game_predictor():
    return render_template('game_predictor.html')

@app.route('/info')
def info():
    return render_template('info.html')
        
if __name__ == '__main__':
    app.run(debug=True)
