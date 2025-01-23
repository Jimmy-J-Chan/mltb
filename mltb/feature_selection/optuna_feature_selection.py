import pandas as pd
from mltb.model_selection.cross_validation import run_cv
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState



def optuna_fs(X, y, cv_g, mdl, metric, cv_task=None, verbose=False):
    """
    Optuna Feature Selection (ofs)
    - TPESampler
    - Hyperband pruner

    Score board
    - tracks the cv scores from each iteration

    :return: df with selected features, score_board (sb)
    """
    if not hasattr(metric, 'greater_is_better'):
        raise AttributeError(f"Metric missing attribute: 'greater_is_better'")
    metric_scaler = 1 if metric.greater_is_better else -1

    sb = pd.DataFrame()
    feats_all = X.columns

    def objective(trial):
        feats = [trial.suggest_categorical(f'f_{c}', [True, False]) for c in feats_all]
        feats_selected = feats_all[feats]

        # Check whether we already evaluated the trial
        states_to_consider = (TrialState.COMPLETE,)
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                # Use the existing value as trial duplicated the parameters.
                return t.value

        oof, score, ytest = run_cv(X[feats_selected], y, None, cv_g, mdl, metric, cv_task, verbose=False)
        score = score * metric_scaler
        return score

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=888), pruner=HyperbandPruner())
    study.optimize(objective, n_trials=500, timeout=60*10)

    # scoreboard and df
    sb = pd.concat([pd.Series(t.params) for t in study.trials], axis=1).T
    sb['score'] = [t.value for t in study.trials]
    sb = sb.drop_duplicates()
    sb.columns = [c[2:] if c.startswith('f_') else c for c in sb.columns]

    best_trial_feats = sb.loc[sb['score'].idxmax()].drop(['score'])
    df_ofs = X[best_trial_feats.loc[best_trial_feats].index]
    return df_ofs, sb

if __name__ == '__main__':
    from sklearn.model_selection import StratifiedKFold
    from lightgbm import LGBMRegressor
    from sklearn.metrics import root_mean_squared_error as rmse

    target = 'price'
    SEED = 888

    train = pd.read_parquet(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\train.parquet")
    test = pd.read_parquet(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\test.parquet")

    all_cols = [c for c in train.columns if c not in [target, 'id','sii']]
    X = train[all_cols]
    y = train[target]
    Xtest = test[all_cols]
    kfolds = 3
    cv_g = StratifiedKFold(n_splits=kfolds)
    metric = rmse
    setattr(metric, 'greater_is_better', False)
    params = {'verbose': -1, 'random_state': SEED}
    mdl = LGBMRegressor(**params)
    cv_task = 'regression'

    # optuna feature selection
    df_ofs, sb = optuna_fs(X, y, cv_g, mdl, metric, cv_task, verbose=True)

    pass
