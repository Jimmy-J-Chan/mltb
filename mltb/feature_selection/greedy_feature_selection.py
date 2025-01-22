import pandas as pd
from mltb.model_selection.cross_validation import run_cv
from mltb.utils.utilities import cc
from itertools import combinations


def gfs(X, y, cv_generator, mdl, metric, cv_task=None, n_features=None, n_features_sort=True, verbose=False):
    """
    Greedy Feature Selection (gfs)
    - gets cv score for all combinations up to n_features
    - if n_features = None, use all features
    - if n_features != None,
        if n_features_sort=True, select top n features sorted by univariate cv score (i.e 1 feature, target)
        if n_features_sort=False, select first n features without sorting
    Score board
    - tracks the cv scores from each iteration

    :return: df with selected features, score_board (sb)
    """
    if not hasattr(metric, 'greater_is_better'):
        raise AttributeError(f"Metric missing attribute: 'greater_is_better'")
    metric_scaler = 1 if metric.greater_is_better else -1

    sb = pd.DataFrame()
    feats_all = X.columns

    # calc univariate cv scores
    print('Calculating univariate cv scores')
    featsN = len(feats_all)-1
    for ix, feat in enumerate(feats_all):
        if verbose:
            print(f' -> iteration: {ix}/{featsN} - {feat}')
        oof, score, ytest = run_cv(X[[feat]], y, None, cv_g, mdl, metric, cv_task, verbose=False)
        score = score * metric_scaler
        tmp_sb = pd.DataFrame({'iteration': ix, 'score': score}, index=[0])
        tmp_sb['features'] = None
        tmp_sb['features'] = tmp_sb['features'].astype(object)
        tmp_sb.at[0, 'features'] = [feat]
        sb = cc(sb, tmp_sb, axis=0)
    sb = sb.reset_index(drop=True)

    feats4comb = feats_all
    if n_features is not None:
        if n_features_sort:
            features_topN = sb.sort_values(by=['score'],ascending=False).head(n_features)
        else:
            features_topN = sb.head(n_features)
        feats4comb = [c[0] for c in features_topN['features']]

    # get all combinations
    feat_all_combs = sum([[list(c) for c in combinations(feats4comb, n)] for n in range(2, len(feats4comb)+1)],[])

    # calc cv scores for all combinations
    print('Calculating cv scores for all combinations')
    featsN = len(feat_all_combs)-1
    itr_offset = sb['iteration'].max()+1
    for ix, feat in enumerate(feat_all_combs):
        if verbose:
            print(f' -> iteration: {ix}/{featsN} - {feat}')
        oof, score, ytest = run_cv(X[feat], y, None, cv_g, mdl, metric, cv_task, verbose=False)
        score = score * metric_scaler
        tmp_sb = pd.DataFrame({'iteration': ix+itr_offset, 'score': score}, index=[0])
        tmp_sb['features'] = None
        tmp_sb['features'] = tmp_sb['features'].astype(object)
        tmp_sb.at[0, 'features'] = feat
        sb = cc(sb, tmp_sb, axis=0)

    # sort score board, best combination at top
    sb = sb.reset_index(drop=True).sort_values(by=['score'], ascending=False)

    # return df with best combination
    df_gfs = X[sb.iloc[0].loc['features']]
    return df_gfs, sb

if __name__ == '__main__':
    from sklearn.model_selection import StratifiedKFold
    from lightgbm import LGBMRegressor
    from sklearn.metrics import root_mean_squared_error as rmse

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
    params = {'verbose': -1, 'random_state': SEED}
    mdl = LGBMRegressor(**params)
    cv_task = 'regression'

    # greedy feature selection
    #df_gfs, sb = gfs(X,y, cv_g, mdl, metric, cv_task, n_features=3, n_features_sort=True, verbose=True)
    df_gfs, sb = gfs(X,y, cv_g, mdl, metric, cv_task, verbose=True)
    # save scoreboard greedy search
    sb.to_pickle(r"C:\Users\n8871191\PycharmProjects\mltb\data\used_car_prices\sb_greedy_feature_selection.pkl")
    pass
