import numpy as np
import pandas as pd
from mltb.model_selection.cross_validation import run_cv
from mltb.utils.utilities import cc

def sfs_backward(X,y, cv_g, mdl, metric, metric_scaler, cv_task, verbose=False):
    lb = pd.DataFrame()
    feats_all = list(X.columns)
    feats_itr = feats_all.copy()

    # inital score
    oof, score, ytest = run_cv(X[feats_itr], y, None, cv_g, mdl, metric, cv_task, verbose=False)
    score = score * metric_scaler
    tmp_lb = pd.DataFrame({'iteration': 'bmk1', 'score': score, 'feat_importance': np.nan, 'feature_dropped': np.nan}, index=[0])
    tmp_lb['features'] = None
    tmp_lb['features'] = tmp_lb['features'].astype(object)
    tmp_lb.at[0, 'features'] = feats_itr
    lb = cc(lb, tmp_lb, axis=0)
    bmk_score = score

    for ix, _ in enumerate(feats_all, 1):
        if len(feats_itr)==1:
            break
        if verbose:
            print(f' -> iteration: {ix}')

        for feat in feats_itr:
            tmp_feats = [c for c in feats_itr if c!=feat]
            oof, score, ytest = run_cv(X[tmp_feats], y, None, cv_g, mdl, metric, cv_task, verbose=False)
            score = score * metric_scaler
            tmp_lb = pd.DataFrame({'iteration': ix, 'score': score, 'feat_importance':bmk_score - score}, index=[0])
            tmp_lb['features'] = None
            tmp_lb['features'] = tmp_lb['features'].astype(object)
            tmp_lb.loc[0,'feature_dropped'] = feat
            tmp_lb.at[0, 'features'] = tmp_feats
            lb = cc(lb, tmp_lb, axis=0)
            pass
        lb = lb.reset_index(drop=True)

        # remove feature that degrades bmk score the most i.e smallest negative feat_importance score
        worst_itr = lb.loc[sb['iteration']==ix].sort_values('feat_importance', ascending=True)
        worst_itr_idx = worst_itr.index[0]
        worst_itr = worst_itr.iloc[0]
        worst_itr_score = worst_itr['score']
        worst_itr_feat = worst_itr['feature_dropped']
        # if removing feature improves score, drop feature, check score > bmk score
        if worst_itr_score > bmk_score:
            feats_itr = [c for c in feats_itr if c!=worst_itr_feat]
            lb.loc[worst_itr_idx, 'worst_iteration'] = True

            # setup next bmkN row in sb
            bmk_lb = lb.loc[[worst_itr_idx]]
            bmk_lb.loc[worst_itr_idx, 'iteration'] = f"bmk{ix+1}"
            bmk_lb.loc[worst_itr_idx, ['feat_importance','feature_dropped','worst_iteration']] = np.nan
            lb = cc(lb, bmk_lb, axis=0).reset_index(drop=True)
            bmk_score = worst_itr_score
        else:
            break

    feats_selection = feats_itr.copy()
    df_sfs = X[feats_selection]
    return df_sfs, lb

def sfs_forward(X,y, cv_g, mdl, metric, metric_scaler, cv_task, verbose=False):
    lb = pd.DataFrame()

    feats_all = X.columns
    feats_selection = []
    best_score_last_itr = 0.
    for ix, _ in enumerate(feats_all, 1):
        if verbose:
            print(f' -> iteration: {ix}')
        feats_itr = feats_all.copy()
        for feat in feats_itr:
            tmp_feats = feats_selection + [feat]
            oof, score, ytest = run_cv(X[tmp_feats], y, None, cv_g, mdl, metric, cv_task, verbose=False)
            score = score * metric_scaler
            tmp_lb = pd.DataFrame({'iteration':ix, 'score': score}, index=[0])
            tmp_lb['features'] = None
            tmp_lb['features'] = tmp_sb['features'].astype(object)
            tmp_lb.at[0,'features'] = tmp_feats
            lb = cc(lb, tmp_lb, axis=0)
        lb = lb.reset_index(drop=True)

        best_itr = lb.loc[lb['iteration']==ix].sort_values('score', ascending=False)
        best_itr_idx = best_itr.index[0]
        best_itr = best_itr.iloc[0]
        best_score_this_itr = best_itr.score
        if (best_score_this_itr > best_score_last_itr) | (ix==1):
            best_score_last_itr = best_score_this_itr
            feats_selection = best_itr.features
            feats_all = [c for c in feats_all if c not in feats_selection]
            lb.loc[best_itr_idx, 'best_iteration'] = True
        else:
            break

    df_sfs = X[feats_selection]
    return df_sfs, lb

def sfs(X, y, cv_generator, mdl, metric, cv_task=None, direction='forward', verbose=False):
    """
    Sequential Feature Selection (sfs)
    forward:
    - start with no features
    - one by one, continue to add features to the model
    - add the feature that gives the highest cv score improvement after each iteration
    - continue until no improvements or select_n_features reached

    backward:
    - start with all features and get cv score
    - one by one, remove features from the model
    - remove the feature that decreases the cv score the most
    - continue until no decreases or select_n_features reached

    Leader board
    - tracks the cv scores from each iteration

    :return: df with selected features, score_board (sb)
    """
    if not hasattr(metric, 'greater_is_better'):
        raise AttributeError(f"Metric missing attribute: 'greater_is_better'")
    metric_scaler = 1 if metric.greater_is_better else -1

    if direction == 'forward':
        df_sfs, lb = sfs_forward(X,y, cv_g, mdl, metric, metric_scaler, cv_task, verbose)
    elif direction == 'backward':
        df_sfs, lb = sfs_backward(X,y, cv_g, mdl, metric, metric_scaler, cv_task, verbose)

    return df_sfs, lb



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
    sfs_direction = 'backward' # 'forward' #

    # sequential feature selection
    df_fw, lb_fw = sfs(X,y, cv_g, mdl, metric, cv_task, 'forward', verbose=True)
    print('forward', list(df_fw.columns.sort_values()))
    df_bw, lb_bw = sfs(X,y, cv_g, mdl, metric, cv_task, 'backward', verbose=True)
    print('backward', list(df_bw.columns.sort_values()))

    # forward ['brand', 'engine', 'fuel_type', 'milage', 'transmission'] -73619.54
    # backward['brand', 'clean_title', 'fuel_type', 'milage', 'model_year', 'transmission'] -73437.73 <- best
    pass


