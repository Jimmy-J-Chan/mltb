import optuna
from mltb.model_selection.cross_validation import run_cv


def hpo_optuna():
    """
    hyperparameter optimization(hpo) using optuna
    - TODO: diversity score from optuna trials - pick top N trails with different diversity
    - TODO: return leaderboard (lb)
    -
    """


    pass





if __name__ == '__main__':
    from lightgbm import LGBMRegressor
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.model_selection import StratifiedKFold

    X = train[all_cols]
    y = train[target]
    Xtest = test[all_cols]
    mdl = LGBMRegressor
    metric = quadratic_weighted_kappa
    params = {'verbose': -1, 'random_state': SEED}
    kfolds = 5
    task = 'regression'

    def objective(trial):
        param_opt = {'learning_rate': trial.suggest_float('learning_rate', 5e-3, 0.1, log=True),
                     'feature_fraction': trial.suggest_float('feature_fraction', 0.4,1),
                     'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 60),
                     'num_leaves': trial.suggest_int('num_leaves', 16, 255),
                     'extra_trees': trial.suggest_categorical('extra_trees', [True,False]),
                     'data_sample_strategy': 'bagging',
                     #'n_estimators': trial.suggest_int('n_estimators', 50, 300, 25),
                     }
        param_opt = param_opt|params
        oof, score, ytest = run_cv(X, y, None, mdl, metric, param_opt, kfolds, task, SEED, verbose=False)

        # with threshold rounding
        def threshold_Rounder(oof_non_rounded, thresholds):
            return np.where(oof_non_rounded < thresholds[0], 0,
                            np.where(oof_non_rounded < thresholds[1], 1,
                                     np.where(oof_non_rounded < thresholds[2], 2, 3)))

        def evaluate_predictions(thresholds, y_true, oof_non_rounded):
            rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
            return -quadratic_weighted_kappa(y_true, rounded_p, map_preds=False)

        tmp_y_true = y.astype(int).clip(0,100)
        ssi_map = ({k:0 for k in range(31)} |
                   {k:1 for k in range(31,50)} |
                   {k:2 for k in range(50,80)} |
                   {k:3 for k in range(80,101)})
        y_sii = tmp_y_true.map(ssi_map)
        KappaOPtimizer = minimize(evaluate_predictions, x0=[31, 50, 80], args=(y_sii, oof), method='Nelder-Mead')
        assert KappaOPtimizer.success, "Optimization did not converge."

        oof_tuned = threshold_Rounder(oof, KappaOPtimizer.x)
        print(f"trial: {trial.number} - {KappaOPtimizer.x}")
        score_tuned = metric(y_sii, oof_tuned)
        return score_tuned

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    best_trail_num = study.best_trial.number
    best_score = study.best_value
    best_params = study.best_params
    print(best_trail_num, best_score, best_params)

    hpo_optuna()
    pass



