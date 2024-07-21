from flask import Flask, render_template
import pandas as pd
from src.mvp_predictor.mvp_pred import mvp_predictions

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/mvp_predictor')
def mvp_predictor():
    """ # Example DataFrame
    data = {
        "Player": ["Player A", "Player B", "Player C"],
        "Points": [30, 25, 20],
        "Assists": [10, 7, 5],
        "Rebounds": [8, 6, 10]
    }
    df = pd.DataFrame(data) """
    
    # Convert DataFrame to HTML
    df = mvp_predictions(2023, 2023)
    columns = df.columns.tolist()
    data = df.values.tolist()

    return render_template('mvp_predictor.html', columns=columns, data=data)

@app.route('/game_predictor')
def game_predictor():
    return "<h1>Game Predictor Page</h1>"

if __name__ == '__main__':
    app.run(debug=True)
