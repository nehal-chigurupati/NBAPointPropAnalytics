from nba_api.stats.endpoints import leagueleaders
import pandas as pd
import plotly.express as px

top_500 = leagueleaders.LeagueLeaders(
    per_mode48='PerGame',
    season='2020-21',
    season_type_all_star='Regular Season',
    stat_category_abbreviation='PTS'
).get_data_frames()[0][:500]

print(top_500)