from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd

def get_player_stats(season):
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed='PerGame')
    stats = player_stats.get_data_frames()[0]
    return stats

def get_player_avg_stats(stats, player_name=None):
    player_avg_stats = stats[['PLAYER_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'TOV', 'PLUS_MINUS']]
    if player_name:
        player_avg_stats = player_avg_stats[player_avg_stats['PLAYER_NAME'].str.contains(player_name, case=False)]
    return player_avg_stats

def get_mvp_winners():
    mvp_winners = {
        '2022-23': 'Joel Embiid',
        '2021-22': 'Nikola Jokic',
        '2020-21': 'Nikola Jokic',
        '2019-20': 'Giannis Antetokounmpo',
        '2018-19': 'Giannis Antetokounmpo',
        '2017-18': 'James Harden',
        '2016-17': 'Russell Westbrook',
        '2015-16': 'Stephen Curry',
        '2014-15': 'Stephen Curry',
        '2013-14': 'Kevin Durant',
        '2012-13': 'LeBron James'
    }
    return mvp_winners

def main():
    start_year = 2012
    end_year = 2023
    seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year + 1)]

    all_stats = {}
    for season in seasons:
        all_stats[season] = get_player_stats(season)

    mvp_winners = get_mvp_winners()

    for year, winner in mvp_winners.items():
        if winner in all_stats[year]['PLAYER_NAME'].values:
            mvp_stats = get_player_avg_stats(all_stats[year], winner)
            print(mvp_stats)
        else:
            print(winner, "stats were not found for the year", year)

    """ player_stats = get_player_stats(seasons[-1])

    #pd.set_option('display.max_columns', None)
    player_avg_stats = get_player_avg_stats(player_stats).sort_values(by='PTS', ascending = False)
    print(player_avg_stats) """

if __name__ == "__main__":
    main()
