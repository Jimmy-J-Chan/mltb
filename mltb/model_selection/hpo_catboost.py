import optuna
from mltb.model_selection.cross_validation import run_cv
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
from mltb.model_selection.get_eval_metric import get_eval_metric
from mltb.utils.utilities import tic, toc
from mltb.model_selection.diversity_selection import diversity_selection

SEED = 888


def hpo_optuna_catboost(X, y, cv_g, mdl, metric, direction=None, cv_task=None, mdl_params=None, n_trials=100, time_limit=None, verbose=True):
    """
    hyperparameter optimization(hpo) for catboost using optuna
    - returns as leaderboard with indication for trials to consider

    optuna pruner
    - https://optuna.readthedocs.io/en/v3.0.2/reference/generated/optuna.integration.CatBoostPruningCallback.html
    - does not work for GPU

    Autogluon search space
    - https://github.com/autogluon/tabrepo/blob/main/tabrepo/models/catboost/generate.py

    time_limit = seconds, time spent running hpo
    """
    if not hasattr(metric, 'greater_is_better'):
        raise AttributeError(f"Metric missing attribute: 'greater_is_better'")
    #metric_scaler = 1 if metric.greater_is_better else -1

    # check X
    cat_cols = [c for c in X.columns if (X[c].dtype=='object')|(X[c].dtype=='category')]
    for c in cat_cols:
        X[c] = X[c].astype('object').fillna('NaN').astype('category')

    def objective(trial):
        params_opt = {
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, step=0.5),
            'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 1, 5),
            'one_hot_max_size': trial.suggest_categorical("one_hot_max_size", [2, 3, 5, 10]),
            'grow_policy': trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
            'use_best_model': True,
            'iterations': 5000,
            'random_strength': 0
        }

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

        eval_metric = get_eval_metric(cv_task).upper()
        mdl_params['eval_metric'] = eval_metric
        mdl_opt = mdl(**params_opt|mdl_params)
        pruning_callback = optuna.integration.CatBoostPruningCallback(trial, eval_metric)
        fit_params = {'callbacks': [pruning_callback],
                      #'verbose': 1000,
                      'early_stopping_rounds': 200,
                      'cat_features': cat_cols,
                      }

        oof, score, ytest = run_cv(X, y, None, cv_g, mdl_opt, metric, cv_task, fit_params=fit_params, verbose=False)

        # evoke pruning manually.
        pruning_callback.check_pruned()

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
    from catboost import CatBoostRegressor
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
    mdl_params = {'verbose': False, 'random_seed': SEED}
    mdl = CatBoostRegressor
    cv_task = 'regression'
    direction = 'minimize'

    lb = hpo_optuna_catboost(X, y, cv_g, mdl, metric, direction, cv_task, mdl_params, n_trials=25, verbose=True)
    lb = hpo_optuna_catboost(X, y, cv_g, mdl, metric, direction, cv_task, mdl_params, time_limit=60*1, verbose=True)
    lb.to_pickle(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\lb_hpo_optuna_catboost.pkl")
    pass



