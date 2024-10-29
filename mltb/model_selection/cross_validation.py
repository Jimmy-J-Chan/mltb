import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def run_cv(X, y, Xtest, mdl, metric, params, kfolds=5, task=None, random_state=888, shuffle=False, verbose=False,
           calc_score=True):
    if not shuffle:
        random_state = None
    if isinstance(Xtest, (pd.DataFrame)):
        if Xtest.empty:
            Xtest = None

    oof = pd.Series()
    ytest = pd.DataFrame()
    kf = StratifiedKFold(n_splits=kfolds, random_state=random_state)
    if verbose: print(f'cv - fold', end='')
    for ix, (train_index, val_index) in enumerate(kf.split(X, y), 1):
        if verbose:
            print(f' {ix}', end='' if ix<kfolds else '\n')

        Xtrain = X.iloc[train_index]
        ytrain = y.iloc[train_index]
        Xval = X.iloc[val_index]

        # train
        mdl_cv = mdl(**params).fit(Xtrain,ytrain)

        # oof/test prediction
        if task in ['regression', 'classification']:
            ypred_oof = pd.Series(mdl_cv.predict(Xval), index=Xval.index)
            ypred_test = pd.Series(mdl_cv.predict(Xtest), index=Xtest.index) if Xtest is not None else pd.Series()
        elif task in ['binary_class']:
            ypred_oof = pd.Series(mdl_cv.predict_proba(Xval)[:,1], index=Xval.index)
            ypred_test = pd.Series(mdl_cv.predict_proba(Xtest)[:,1], index=Xtest.index) if Xtest is not None else pd.Series()
        elif task in ['proba']:
            ypred_oof = pd.DataFrame(mdl_cv.predict_proba(Xval), index=Xval.index)
            ypred_test = pd.DataFrame(mdl_cv.predict_proba(Xtest), index=Xtest.index) if Xtest is not None else pd.Series()

        oof = pd.concat([oof, ypred_oof], axis=0)
        ytest = pd.concat([ytest, ypred_test], axis=1)
        pass

    if not ytest.empty:
        if task in ['proba']:
            ytest = ytest.groupby(by=ytest.columns, axis=1).mean()
        else:
            ytest = ytest.mean(axis=1)

    oof = oof.reindex(y.index)
    if task in ['proba']:
        return oof, ytest
    else:
        score = None
        if calc_score:
            score = metric(y, oof)
        return oof, score, ytest


def cv(X, y, mdl, metric, params, kfolds=5, random_state=888, shuffle=False):
    if not shuffle:
        random_state = None

    oof = pd.Series()
    kf = KFold(n_splits=kfolds, random_state=random_state)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        Xtrain = X.iloc[train_index]
        ytrain = y.iloc[train_index]
        Xtest = X.iloc[test_index]
        #ytest = y.iloc[test_index]

        mdl_cv = mdl(**params).fit(Xtrain,ytrain)
        ypred = pd.Series(mdl_cv.predict(Xtest), index=Xtest.index)
        oof = pd.concat([oof, ypred], axis=0)
        pass

    oof = oof.reindex(y.index)
    score = metric(y, oof)
    return oof, score

