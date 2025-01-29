import optuna
from mltb.model_selection.cross_validation import run_cv
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
from mltb.model_selection.get_eval_metric import get_eval_metric
from mltb.utils.utilities import tic, toc
from mltb.model_selection.diversity_selection import diversity_selection

SEED = 888

def hpo_optuna_xgboost(X, y, cv_g, mdl, metric, direction=None, cv_task=None, mdl_params=None, n_trials=100, time_limit=None, verbose=True):
    """
    hyperparameter optimization(hpo) for xgboost using optuna
    - returns as leaderboard with indication for trials to consider

    optuna pruner
    - https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_integration.py

    # https://rdrr.io/github/Laurae2/LauraeDS/man/Laurae.xgb.train.html

    autogluon search space
    - https://github.com/autogluon/tabrepo/blob/main/tabrepo/models/xgboost/generate.py

    time_limit = seconds, time spent running hpo
    """
    if not hasattr(metric, 'greater_is_better'):
        raise AttributeError(f"Metric missing attribute: 'greater_is_better'")
    #metric_scaler = 1 if metric.greater_is_better else -1
    cat_cols = [c for c in X.columns if (X[c].dtype=='object')|(X[c].dtype=='category')]

    def objective(trial):
        params_opt = {
            'booster': trial.suggest_categorical("booster", ["gbtree", "dart"]),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 1.5, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
            'early_stopping_rounds': 100,
        }

        if (params_opt["booster"] == "gbtree") | (params_opt["booster"] == "dart"):
            params_opt["max_depth"] = trial.suggest_int("max_depth", 1, 10)
            params_opt["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            params_opt["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            params_opt["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if params_opt["booster"] == "dart":
            params_opt["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            params_opt["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            params_opt["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            params_opt["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        if len(cat_cols)>0:
            params_opt['enable_categorical'] = True

        # manage time_limit
        if time_limit is not None:
            secs_elapsed = toc(secs_elapsed=True)
            if secs_elapsed > time_limit:
                print('EXIT STUDY...time limited exceeded')
                return study.stop()

        # Check whether we already evaluated the trial
        states_to_consider = (TrialState.COMPLETE,)
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value # Use the existing value as trial duplicated the parameters.

        eval_metric = get_eval_metric(cv_task)
        other_params = {'callbacks': [optuna.integration.XGBoostPruningCallback(trial, f"validation_0-{eval_metric}")],
                        'eval_metric': metric}
        fit_params = {'verbose': mdl_params['verbose']}
        mdl_opt = mdl(**params_opt|mdl_params|other_params)

        oof, score, ytest = run_cv(X, y, None, cv_g, mdl_opt, metric, cv_task, fit_params=fit_params,verbose=False)
        #score = score * metric_scaler
        return score

    if time_limit is not None:
        tic()
        n_trials = 10_000

    study = optuna.create_study(direction=direction, sampler=TPESampler(seed=SEED), pruner=HyperbandPruner())
    study.optimize(objective, n_trials=n_trials, timeout=60*10)

    # check n_trials are conducted (sometimes exits early), or time_limit not reached yet
    if time_limit is not None:
        secs_elapsed = toc(secs_elapsed=True)
        while secs_elapsed < time_limit:
            study.optimize(objective, n_trials=n_trials, timeout=60*10)
            secs_elapsed = toc(secs_elapsed=True)
    else:
        trials2complete = n_trials - len(study.trials)
        while trials2complete > 0:
            study.optimize(objective, n_trials=trials2complete, timeout=60*10)
            trials2complete = trials - len(study.trials)

    # leaderboard and df
    lb = study.trials_dataframe(attrs=['value','params']).rename(columns={'value':'score'}).dropna(subset=['score'])
    if len(lb) > 0:
        lb = lb.drop_duplicates()
        lb.columns = [c[7:] if c.startswith('params_') else c for c in lb.columns]
        lb = lb.sort_values(by=['score'], ascending=False if metric.greater_is_better else True)

        # diversity selection
        trials2keep = diversity_selection(lb)
        lb['trials2keep'] = False
        lb.loc[lb.index.isin(trials2keep),'trials2keep'] = True
    return lb


if __name__ == '__main__':
    import pandas as pd
    from xgboost import XGBRegressor
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.model_selection import StratifiedKFold

    target = 'price'
    SEED = 888

    # train = pd.read_parquet(r"C:\Users\n8871191\PycharmProjects\mltb\data\used_car_prices\train.parquet")
    # test = pd.read_parquet(r"C:\Users\n8871191\PycharmProjects\mltb\data\used_car_prices\test.parquet")
    train = pd.read_parquet(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\train.parquet")
    test = pd.read_parquet(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\test.parquet")

    all_cols = [c for c in train.columns if c not in [target, 'id','sii']]
    X = train[all_cols]#.sample(5000)
    y = train[target]#.loc[X.index]
    Xtest = test[all_cols]
    kfolds = 3
    cv_g = StratifiedKFold(n_splits=kfolds)
    metric = rmse
    setattr(metric, 'greater_is_better', False)
    mdl_params = {'verbose': False, 'random_state': SEED}
    mdl = XGBRegressor
    cv_task = 'regression'
    direction = 'minimize'

    lb = hpo_optuna_xgboost(X, y, cv_g, mdl, metric, direction, cv_task, mdl_params, n_trials=20, verbose=True)
    # lb = hpo_optuna_xgboost(X, y, cv_g, mdl, metric, direction, cv_task, mdl_params, time_limit=60*1, verbose=True)
    lb.to_pickle(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\lb_hpo_optuna_xgboost.pkl")
    pass



