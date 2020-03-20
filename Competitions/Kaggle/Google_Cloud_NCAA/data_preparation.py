import numpy as np
import pandas as pd


def prepare_template(df):
    df_winners = df[['Season', 'WTeamID', 'LTeamID']].rename(columns={'WTeamID': 'TeamID_1', 'LTeamID': 'TeamID_2'})
    df_winners['is_won'] = 1
    df_losers = df[['Season', 'LTeamID', 'WTeamID']].rename(columns={'LTeamID': 'TeamID_1', 'WTeamID': 'TeamID_2'})
    df_losers['is_won'] = 0
    return pd.concat([df_winners, df_losers])


def prepare_seeds(seeds_df):
    df = seeds_df.copy()
    df['SeedNum'] = df['Seed'].apply(lambda x: int(x[1:3]))
    return df.set_index(['Season', 'TeamID']).drop(columns=['Seed'])


def flatten_compact_results(df):
    df = df.copy()
    df['ScoreDiff'] = df['WScore'] - df['LScore']
    win_df = df[['Season', 'WTeamID', 'WScore', 'ScoreDiff']].rename(columns={'WTeamID': 'TeamID', 'WScore': 'Score'})
    lose_df = df[['Season', 'LTeamID', 'LScore', 'ScoreDiff']].rename(columns={'LTeamID': 'TeamID', 'LScore': 'Score'})
    lose_df['ScoreDiff'] = -lose_df['ScoreDiff']
    return pd.concat([win_df, lose_df], axis=0).reset_index(drop=True)


def prepare_regular_season_results(regseason_results: pd.DataFrame) -> pd.DataFrame:
    def wincount(x):
        return np.sum(x > 0)

    regseason_flatten_results = flatten_compact_results(regseason_results)
    df = regseason_flatten_results.groupby(['Season', 'TeamID']).agg(
        {
            'Score': ['sum', 'mean', 'median', 'max', 'min'],
            'ScoreDiff': ['sum', 'mean', 'median', 'max', 'min', 'count', wincount]
        }
    )
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df['WinRate'] = df['ScoreDiff_wincount'] / df['ScoreDiff_count']
    return df.drop(columns=['ScoreDiff_count', 'ScoreDiff_wincount'])


def calculate_features(df, feature_names, name_func):
    name, func = name_func
    for fname in feature_names:
        df[f'{fname}_{name}'] = func((df[f'{fname}_1'], df[f'{fname}_2']))
    return df


def merge_data(template: pd.DataFrame, data: dict):
    df = template
    suffixes = ('_1', '_2')

    # add seeds
    seeds_df = prepare_seeds(data['Tourney']['Seeds'])
    df = pd.merge(df, seeds_df, how='left', left_on=['Season', 'TeamID_1'],
                  right_on=['Season', 'TeamID'], suffixes=suffixes)
    df = pd.merge(df, seeds_df, how='left', left_on=['Season', 'TeamID_2'],
                  right_on=['Season', 'TeamID'], suffixes=suffixes)

    # add regular season results
    regseasons_results = prepare_regular_season_results(data['RegularSeason']['CompactResults'])
    df = pd.merge(df, regseasons_results, how='left', left_on=['Season', 'TeamID_1'],
                  right_on=['Season', 'TeamID'], suffixes=suffixes)
    df = pd.merge(df, regseasons_results, how='left', left_on=['Season', 'TeamID_2'],
                  right_on=['Season', 'TeamID'], suffixes=suffixes)

    func = ('diff', lambda x: x[0] - x[1])
    features = ['SeedNum',
                'WinRate', 'Score_sum', 'Score_mean', 'Score_median']
    df = calculate_features(df, features, func)

    func = ('ratio', lambda x: x[0] / x[1])
    features = [
        'WinRate', 'Score_min', 'Score_max', 'Score_mean', 'Score_median']
    df = calculate_features(df, features, func)

    return df


def prepare_test(submission_df):
    def parse_id(x: str):
        return list(map(int, x.split('_')))

    df = submission_df
    df['Season'], df['TeamID_1'], df['TeamID_2'] = zip(*df['ID'].map(parse_id))
    return df
