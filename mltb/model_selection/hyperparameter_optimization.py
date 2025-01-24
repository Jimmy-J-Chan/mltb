import optuna
from mltb.model_selection.cross_validation import run_cv
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState

SEED = 888

def hpo_optuna_lgbm(X, y, cv_g, mdl, metric, cv_task, mdl_params=None, verbose=True):
    """
    hyperparameter optimization(hpo) using optuna
    - TODO: diversity score from optuna trials - pick top N trails with different diversity
    - TODO: return leaderboard (lb)
    -
    """
    if not hasattr(metric, 'greater_is_better'):
        raise AttributeError(f"Metric missing attribute: 'greater_is_better'")
    metric_scaler = 1 if metric.greater_is_better else -1

    def objective(trial):
        # https://rdrr.io/github/Laurae2/LauraeDS/man/Laurae.xgb.train.html
        params_opt = {
            'boosting_type': trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            'max_bin': trial.suggest_int('max_bin', 64, 512),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=.01),
            'num_iterations': trial.suggest_int('num_iterations', 50, 500, step=50),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'max_depth ': -1, # unlimited
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'extra_trees': trial.suggest_categorical("extra_trees", [True, False]),
        }

        # Check whether we already evaluated the trial
        states_to_consider = (TrialState.COMPLETE,)
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value # Use the existing value as trial duplicated the parameters.

        mdl_opt = mdl(**params_opt|mdl_params)
        oof, score, ytest = run_cv(X, y, None, cv_g, mdl_opt, metric, cv_task, verbose=False)
        score = score * metric_scaler
        return score

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED), pruner=HyperbandPruner())
    study.optimize(objective, n_trials=250, timeout=60*10)

    # leaderboard and df
    lb = study.trials_dataframe(attrs=['value','params']).rename(columns={'value':'score'})
    lb = lb.drop_duplicates()
    lb.columns = [c[7:] if c.startswith('params_') else c for c in lb.columns]
    lb = lb.sort_values(by=['score'], ascending=False)

    # diversity selection

    return lb





if __name__ == '__main__':
    import pandas as pd
    from lightgbm import LGBMRegressor
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.model_selection import StratifiedKFold

    target = 'price'
    SEED = 888

    train = pd.read_parquet(r"C:\Users\n8871191\PycharmProjects\mltb\data\used_car_prices\train.parquet")
    test = pd.read_parquet(r"C:\Users\n8871191\PycharmProjects\mltb\data\used_car_prices\test.parquet")

    all_cols = [c for c in train.columns if c not in [target, 'id','sii']]
    X = train[all_cols]
    y = train[target]
    Xtest = test[all_cols]
    kfolds = 3
    cv_g = StratifiedKFold(n_splits=kfolds)
    metric = rmse
    setattr(metric, 'greater_is_better', False)
    mdl_params = {'verbose': -1, 'random_state': SEED}
    mdl = LGBMRegressor
    cv_task = 'regression'

    lb = hpo_optuna_lgbm(X, y, cv_g, mdl, metric, cv_task, mdl_params, verbose=True)
    lb.to_pickle(r"C:\Users\n8871191\PycharmProjects\mltb\data\used_car_prices\lb_hpo_optuna_lgbm.pkl")
    pass



