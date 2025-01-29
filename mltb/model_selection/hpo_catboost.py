import optuna
from mltb.model_selection.cross_validation import run_cv
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
from mltb.model_selection.get_eval_metric import get_eval_metric
from mltb.utils.utilities import tic, toc

SEED = 888

def diversity_selection(lb=None):
    from sklearn.metrics.pairwise import cosine_similarity
    #lb = pd.read_pickle(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\lb_hpo_optuna_lgbm.pkl")

    feats = [c for c in lb.columns if c!='score']
    lb_calc = lb.copy()

    # encode
    bool_cols = [c for c in lb_calc.columns if lb_calc[c].dtype=='bool']
    lb_calc.loc[:,bool_cols] = lb_calc.loc[:,bool_cols].astype(int)
    cat_cols = [c for c in lb_calc.columns if (lb_calc[c].dtype=='object')|(lb_calc[c].dtype=='category')]
    for cat_col in cat_cols:
        cmap = {c:ix for ix, c in enumerate(lb_calc[cat_col].unique())}
        lb_calc.loc[:,cat_col] = lb_calc.loc[:,cat_col].map(cmap)

    lb_calc = lb_calc[feats].head(int(0.33*len(lb))) # pick top 33%
    sim = pd.DataFrame(cosine_similarity(lb_calc), index=lb_calc.index, columns=lb_calc.index)

    # iteratively pick trials that are diverse in hyperparamters as measured using cosine similarity
    best_trial = lb_calc.index[0]
    trials2keep = [best_trial]
    threshold = 0.7
    for trial in sim.columns:
        tmp = sim.loc[trials2keep, trial] # sim scores between trials already picked against potential trial to be added
        if (tmp<=threshold).all(): # if all trials picked < threshold against trial to be added, add to trials to keep
            trials2keep.append(trial)

    return trials2keep

def hpo_optuna_catboost(X, y, cv_g, mdl, metric, direction=None, cv_task=None, mdl_params=None, n_trials=100, time_limit=None, verbose=True):
    """
    hyperparameter optimization(hpo) for catboost using optuna
    - returns as leaderboard with indication for trials to consider

    optuna pruner
    - https://optuna.readthedocs.io/en/v3.0.2/reference/generated/optuna.integration.CatBoostPruningCallback.html
    - does not work for GPU

    # https://rdrr.io/github/Laurae2/LauraeDS/man/Laurae.xgb.train.html

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
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=.01),
            'depth': trial.suggest_int('depth', 1, 16),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10, step=0.5),
            'use_best_model': True,
            'iterations': 5000,
            'random_strength': 0
        }

        # manage time_limit
        if time_limit is not None:
            secs_elapsed = toc(secs_elapsed=True)
            if secs_elapsed > time_limit:
                print('EXIT STUDY...time limited surpassed')
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

    lb = hpo_optuna_catboost(X, y, cv_g, mdl, metric, direction, cv_task, mdl_params, n_trials=200, verbose=True)
    lb = hpo_optuna_catboost(X, y, cv_g, mdl, metric, direction, cv_task, mdl_params, time_limit=60*1, verbose=True)
    lb.to_pickle(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\lb_hpo_optuna_catboost.pkl")
    pass



