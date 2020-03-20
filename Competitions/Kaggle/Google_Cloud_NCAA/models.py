import numpy as np

from sklearn.metrics import log_loss


def prepare_for_lgbm(df):
    df = df.drop(columns=['TeamID_1', 'TeamID_2'])
    return df


def train_test_split(train_df, test_df, year=2019):
    train = train_df[train_df['Season'] < year].drop(columns=['Season'])
    test = test_df[test_df['Season'] == year].drop(columns=['Season'])
    X_train, X_test = train.drop(columns=['is_won']), test
    y_train = train['is_won']
    return X_train, X_test, y_train


def train_val_split(df, year=2019):
    train = df[df['Season'] < year].drop(columns=['Season'])
    validation = df[df['Season'] == year].drop(columns=['Season'])
    X_train, X_val = train.drop(columns=['is_won']), validation.drop(columns=['is_won']),
    y_train, y_val = train['is_won'], validation['is_won']
    return X_train, X_val, y_train, y_val


def predict_test(train_df, test_df, model, fit_params=None):
    predictions = []
    train_df = prepare_for_lgbm(train_df)
    test_df = prepare_for_lgbm(test_df)
    for year in np.arange(2015, 2020):
        X_train, X_test, y_train = train_test_split(train_df, test_df, year=year)
        print(year, X_train.shape, X_test.shape, y_train.shape)
        model.fit(X_train, y_train)
        y_pred_test = model.predict_proba(X_test)[:, 1]
        predictions.append(y_pred_test)
    return np.hstack(predictions)


def validate_model(df, model, fit_params=None, verbose=False):
    ys = {'y_train': [], 'y_val': [], 'y_train_preds': [], 'y_val_preds': []}
    for year in np.arange(2015, 2020):
        X_train, X_val, y_train, y_val = train_val_split(prepare_for_lgbm(df), year=year)
        if verbose:
            print(year, X_train.shape, X_val.shape, y_train.shape, y_val.shape)
        model.fit(X_train, y_train)
        y_pred_train = model.predict_proba(X_train)[:, 1]
        y_pred_val = model.predict_proba(X_val)[:, 1]
        ys['y_train'].append(y_train)
        ys['y_val'].append(y_val)
        ys['y_train_preds'].append(y_pred_train)
        ys['y_val_preds'].append(y_pred_val)

    score_train = log_loss(np.hstack(ys['y_train']), np.hstack(ys['y_train_preds']))
    score_val = log_loss(np.hstack(ys['y_val']), np.hstack(ys['y_val_preds']))
    if verbose:
        print(score_train, score_val)
    return score_val
