#API and numerical imports
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playercareerstats, leaguegamefinder, playergamelog
from nba_api.stats.library.parameters import Season, SeasonType
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import csv 

#Machine learning imports
from sklearn.model_selection import train_test_split
import sklearn
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler



#default days of rest set for the first game of the season:
first_game_rest = 7

#default average ppg since start of season for player
default_first_game_ppg = 0.0

def correct_id(abbrev):
    if abbrev == "NJN":
        return "BKN"
    elif abbrev == "NOH" or abbrev == "NOK":
        return "NOP"
    elif abbrev == "SEA":
        return "OKC"
    elif abbrev == "PHO":
        return "PHX"
    elif abbrev == "GOS":
        return "GSW"
    else:
        return abbrev
    
def get_team_id_from_abbrev(abbrev):
    abbrev = correct_id(abbrev)
    return [t for t in teams.get_teams() if t['abbreviation'] == abbrev][0]['id']

months = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12
}
def convert_human_date(d):
    month = months[d[0:3]]
    day = int(d[4:6])
    year = int(d[8:])

    return date(year, month, day)

def distance_between_dates(d1, d2):
    return int((d2 - d1).days)


def make_prediction(player_full_name, point_threshold, player_abbrev, opp_abbrev, game_date):
    #Retrieve player info
    player_details = [p for p in players.get_players() if p["full_name"] == player_full_name][0]

    #Retrieve all the seasons the player has been in the league
    career_stats = playercareerstats.PlayerCareerStats(player_id=player_details['id']).get_data_frames()[0]
    seasons = career_stats['SEASON_ID']

    #Fix type of season
    season_type = SeasonType.regular

    #Retrieve game-by-game stats for player 
    season_data = {}
    for s in seasons:
        season_data[s] = playergamelog.PlayerGameLog(player_id=player_details['id'], season=s, season_type_all_star=season_type).get_data_frames()[0]
    
    """
    For each game, we want to extract the following information for training:
    1. Opponent
    2. Player's team
    3. Days since last game
    4. Game number
    5. Average PPG since start of season
    """

    #Form training arrays
    #Entries in each tuple agree with the above
    X = []
    #Y will be 1 if player scores above threshold, 0 if below
    Y = []

    #Label training data
    for season in season_data.keys():
        currSeason = season_data[season]
        for ind in currSeason.index:
            if float(currSeason['PTS'][ind]) > float(point_threshold):
                Y.append(1)
            else:
                Y.append(0)
    

    #Collect player team and opponent team IDs for training data
    player_team_ids = []
    opp_team_ids = []
    for season in season_data.keys():
        currSeason = season_data[season]
        for ind in currSeason.index:
            matchup = str(currSeason['MATCHUP'][ind])
            player_team_abbrev = matchup[0:3]

            #@ index
            if "@" in matchup:
                opp_team_abbrev = matchup[6:]
            elif "vs." in matchup:
                opp_team_abbrev = matchup[8:]
            else:
                raise Exception("Unconforming matchup string: " + matchup)

            player_team_ids.append(get_team_id_from_abbrev(player_team_abbrev))
            opp_team_ids.append(get_team_id_from_abbrev(opp_team_abbrev))
    
    days_since_last_game = []
    for season in season_data.keys():
        currSeason = season_data[season]
        for ind in currSeason.index:
            if (ind + 1) == len(currSeason.index):
                days_since_last_game.append(first_game_rest)
            else:
                currGameDate = convert_human_date(str(currSeason["GAME_DATE"][ind]))
                lastGameDate = convert_human_date(str(currSeason["GAME_DATE"][ind+1]))
                days_since_last_game.append(distance_between_dates(lastGameDate, currGameDate))
    
    #Get the game number, in terms of the number of games that player has played in the season
    game_numbers = []
    for season in season_data.keys():
        for ind in season_data[season].index:
            game_numbers.append(len(season_data[season].index) - ind)
    
    #Compute average ppg since start of season
    average_ppg = []
    for season in season_data.keys():
        currSeason = season_data[season]
        for ind in currSeason.index:
            if (ind + 1) == len(currSeason.index):
                average_ppg.append(default_first_game_ppg)
            else:
                sum = 0.0
                for pts in currSeason["PTS"][ind+1:len(currSeason.index)]:
                    sum += float(pts)
                average_ppg.append(sum / (len(currSeason.index) - ind - 1))
        
    #Zip together all the training input data
    for i in range(len(game_numbers)):
        X.append([float(player_team_ids[i]), float(opp_team_ids[i]), float(days_since_last_game[i]), float(game_numbers[i]), average_ppg[i]])
    
    """
    Below we start doing preprocessing
    """

    Y = pd.DataFrame(Y)
    X = pd.DataFrame(X)
    
    scaler = MinMaxScaler()
    scaler.fit(X)
    t_X = scaler.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(t_X, Y, test_size=0.25, random_state=0)

    basic_model = Sequential()

    basic_model.add(Dense(units=60, activation='relu', input_shape=(5,)))
    basic_model.add(Dense(units=40, activation='relu'))
    basic_model.add(Dense(units=20, activation='sigmoid'))
    basic_model.add(Dense(1, activation='sigmoid'))

    adam = keras.optimizers.legacy.Adam(learning_rate=0.001)
    basic_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    basic_model.fit(X_train, Y_train, epochs=100, verbose=0)

    loss_and_metrics = basic_model.evaluate(X_test, Y_test)

    ret = {}

    ret["accuracy"] = loss_and_metrics[1]

    opp_id, player_id = get_team_id_from_abbrev(opp_abbrev), get_team_id_from_abbrev(player_abbrev)
    days_elapsed = (convert_human_date(game_date) - convert_human_date(season_data["2023-24"]["GAME_DATE"][0])).days
    game_number = len(season_data["2023-24"].index) + 1
    average_ppg = 0.0
    currSeason = season_data['2023-24']
    for ind in currSeason.index:
            if (ind + 1) == len(currSeason.index):
                average_ppg = (default_first_game_ppg)
            else:
                sum = 0.0
                for pts in currSeason["PTS"][ind+1:len(currSeason.index)]:
                    sum += float(pts)
                average_ppg = (sum / (len(currSeason.index) - ind - 1))

    input = pd.DataFrame([[float(player_id), float(opp_id), float(days_elapsed), float(game_number), float(average_ppg)]])
    ret["prediction"] = basic_model.predict(input, verbose=0)[0][0]
    return ret

def get_command_line_input():
    numThresholds = input("\033[1mEnter number of thresholds:\033[0m ")
    bets = []
    for i in range(len(numThresholds)):
        name = input("\033[1mBet" + str(i)+ " player name:\033[0m ")
        line = input("\033[1mBet" + str(i)+ " point threshold:\033[0m ")
        playerteam = input("\033[1mBet" + str(i)+ " player's team abbreviation:\033[0m ")
        oppteam = input("\033[1mBet" + str(i)+ " opposing team abbreviation:\033[0m ")
        gamedate = input("\033[1mBet" + str(i)+ " game date:\033[0m ")
        bets.append([name, line, playerteam, oppteam, gamedate])


    results = []
    for i in bets:
        print("\033[1mResult of bet " + str(i) + ":\033[0m ")
        print(make_prediction(i[0], i[1], i[2], i[3], i[4]))

def get_csv_input():
    print("\033[1mGetting csv input.\033[0m")
    file = open("bets.csv", encoding='utf-8-sig')
    csvreader = csv.reader(file)
    bets = []
    for row in csvreader:
        bets.append(row)
    
    for i in bets:
        prediction = make_prediction(i[0], i[1], i[2], i[3], i[4])
    
    for i in bets:
        print("\033[1mResult of bet " + str(i) + ":\033[0m ")
        print(prediction)
    
print(make_prediction("Michael Porter Jr.", 17.5, "DEN", "NOP", "NOV 17, 2023"))
