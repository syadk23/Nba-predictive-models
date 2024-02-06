from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd

def get_player_stats(season):
    # Fetching player stats for the specified season
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed='PerGame')
    stats = player_stats.get_data_frames()[0]
    return stats

def get_player_avg_stats(stats):
    # Extracting relevant columns
    avg_stats = stats[['PLAYER_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'TOV', 'PLUS_MINUS']]
    return avg_stats

def main():
    season = '2023-24'

    stats = get_player_stats(season)

    #pd.set_option('display.max_columns', None)
    avg_stats = get_player_avg_stats(stats).sort_values(by='PTS', ascending = False)
    print(avg_stats)

if __name__ == "__main__":
    main()
