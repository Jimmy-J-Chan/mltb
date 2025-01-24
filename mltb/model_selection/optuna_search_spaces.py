import optuna

"""
LGBM

For Better Accuracy
Use large max_bin (may be slower)
Use small learning_rate with large num_iterations
Use large num_leaves (may cause over-fitting)
Use bigger training data
Try dart

Deal with Over-fitting
Use small max_bin
Use small num_leaves
Use min_data_in_leaf and min_sum_hessian_in_leaf
Use bagging by set bagging_fraction and bagging_freq
Use feature sub-sampling by set feature_fraction
Use bigger training data
Try lambda_l1, lambda_l2 and min_gain_to_split for regularization
Try max_depth to avoid growing deep tree
Try extra_trees
Try increasing path_smooth
"""

ss_lgbm = {
        'boosting_type': trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }


