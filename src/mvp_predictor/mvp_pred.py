from nba_api.stats.endpoints import leaguedashplayerstats

def get_player_stats(season):
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed='PerGame')
    stats = player_stats.get_data_frames()[0]
    return stats

def get_player_advanced_stats(season):
    player_advanced_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame')
    advanced_stats = player_advanced_stats.get_data_frames()[0]
    return advanced_stats

def get_player_avg_stats(stats, player_name=None):
    player_avg_stats = stats[['PLAYER_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'TOV', 'PLUS_MINUS']]
    if player_name:
        player_avg_stats = player_avg_stats[player_avg_stats['PLAYER_NAME'].str.contains(player_name, case=False)]
    return player_avg_stats

def get_mvp_winners():
    # Use last 10 as testing for now
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
