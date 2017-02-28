import sqlite3
import warnings

import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import model_selection

warnings.simplefilter("ignore")

path = "/home/omar/mine/coursework1/code/"  # Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

HOME_PLAYERS_STR = ["home_player_" + str(x) for x in range(1, 12)]
AWAY_PLAYERS_STR = ["away_player_" + str(x) for x in range(1, 12)]

PLAYERS_COLUMNS = [
        'player_api_id', 'date', 'overall_rating', 'potential', 'attacking_work_rate',
        'defensive_work_rate', 'crossing', 'finishing', 'heading_accuracy', 'short_passing',
        'volleys', 'dribbling', 'curve', 'free_kick_accuracy', 'long_passing', 'ball_control',
        'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power',
        'jumping', 'stamina', 'strength', 'long_shots', 'aggression', 'interceptions',
        'positioning', 'vision', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
        'gk_reflexes'
    ]

def get_match_data():
    interesting_columns = [
        "date", "match_api_id", "home_team_api_id",
        "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
        "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
        "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"
    ]

    columns_not_null = HOME_PLAYERS_STR + AWAY_PLAYERS_STR
    not_null_condition = " and ".join(x + " is not null" for x in columns_not_null)

    match_data = pd.read_sql("SELECT " + ','.join(interesting_columns) + " FROM Match WHERE "
                             + not_null_condition + ";", conn)
    return match_data


def get_player_attributes():
    players = pd.read_sql(
        "SELECT " + ','.join(PLAYERS_COLUMNS) + " FROM player_attributes;", conn)
    return players


def get_team_quality_by_team(team_members, prefix):
    converter = {
        "low": 1,
        "medium": 2,
        "high": 3
    }
    team_members = team_members.drop('player_api_id', 1).drop('date', 1)
    column_names = team_members.columns.values
    qualities = []
    for col in column_names:
        if col == 'attacking_work_rate' or col == 'defensive_work_rate':
            qual = [converter[x] for x in team_members[col] if x in converter.keys()]
            qualities.append(sum(qual) / len(qual))
        elif col.startswith("gk"):
            qualities.append(max(team_members[col]))
        else:
            qualities.append(team_members[col].median())
    return pd.Series(qualities, [prefix + x for x in column_names])


def get_team_quality(match, players, team_members_names, prefix=''):
    team_members = pd.DataFrame()
    for player in team_members_names:
        player_id = match[player]
        player_stats = players[players.player_api_id == player_id]
        player_info = player_stats[player_stats.date <
                                             match['date']].sort_values(
            by='date', ascending=True).iloc[0]
        team_members = team_members.append(player_info, ignore_index=True)
    team_vector = get_team_quality_by_team(team_members, prefix=prefix)
    return team_vector


def liberate_data(f):
    def closure(*args, **kwargs):
        return f(*args, **kwargs).dropna()
    return closure


@liberate_data
def build_data(dump_file=None):
    if dump_file is not None:
        return pd.read_pickle(dump_file)
    match_data = get_match_data()
    players = get_player_attributes()
    data = pd.DataFrame()

    for i, match in match_data.iterrows():
        home_team_quality = get_team_quality(match, players, HOME_PLAYERS_STR, prefix='home_')
        away_team_quality = get_team_quality(match, players, AWAY_PLAYERS_STR, prefix='away_')
        match_result = match['home_team_goal'] - match['away_team_goal']
        result = 0 if match_result < 0 else 1 if match_result == 0 else 2
        match_info = pd.concat((pd.Series([result], ['result']),
                                home_team_quality, away_team_quality))
        data = data.append(match_info, ignore_index=True)

    return data


def split_data(data):
    inp = data.iloc[:, :-1]
    out = data.iloc[:, -1:].astype(int)
    return inp, out


def get_logistic_regr_score(X_train, Y_train, X_test, Y_test):
    log_regr = linear_model.LogisticRegression(solver='newton-cg', random_state=42)
    log_regr.fit(X_train, Y_train)
    return log_regr.score(X_test, Y_test)


def get_random_forest_score(X_train, Y_train, X_test, Y_test):
    random_forest = ensemble.RandomForestClassifier(n_estimators=400, random_state=42)
    random_forest.fit(X_train, Y_train)
    return random_forest.score(X_test, Y_test)


def get_ada_boost_score(X_train, Y_train, X_test, Y_test):
    ada_boost = ensemble.AdaBoostClassifier(n_estimators=400, random_state=42)
    ada_boost.fit(X_train, Y_train)
    return ada_boost.score(X_test, Y_test)


def get_gaussian_score(X_train, Y_train, X_test, Y_test):
    gaussian = naive_bayes.GaussianNB()
    gaussian.fit(X_train, Y_train)
    return gaussian.score(X_test, Y_test)


def get_kneighbor_score(X_train, Y_train, X_test, Y_test):
    kneighbor = neighbors.KNeighborsClassifier(n_neighbors=200, algorithm='brute')
    kneighbor.fit(X_train, Y_train)
    return kneighbor.score(X_test, Y_test)


def main():
    data = build_data(dump_file="my_match_data")
    X, Y = split_data(data)
    print("Data shape. X:", X.shape, "Y:", Y.shape)
    X_train, X_test, Y_train, Y_test = \
        model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

    log_regr_res = get_logistic_regr_score(X_train, Y_train, X_test, Y_test)
    print("Logistic Regression score:", log_regr_res)

    random_forest_res = get_random_forest_score(X_train, Y_train, X_test, Y_test)
    print("Random Forest score:", random_forest_res)

    ada_boost_res = get_ada_boost_score(X_train, Y_train, X_test, Y_test)
    print("Ada Boost score:", ada_boost_res)

    gaussian_res = get_gaussian_score(X_train, Y_train, X_test, Y_test)
    print("GaussianNB score:", gaussian_res)

    kneighbor_res = get_kneighbor_score(X_train, Y_train, X_test, Y_test)
    print("KNeighbor score:", kneighbor_res)


if __name__ == "__main__":
    main()
