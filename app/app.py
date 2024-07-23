from flask import Flask, render_template, session, redirect, url_for
from src.mvp_predictor.mvp_pred import mvp_predictions
from datetime import datetime
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

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

    return render_template('mvp_predictor.html', selected_year=year-1, columns=columns, data=data)

@app.route('/game_predictor')
def game_predictor():
    return "<h1>Game Predictor Page</h1>"

if __name__ == '__main__':
    app.run(debug=True)
