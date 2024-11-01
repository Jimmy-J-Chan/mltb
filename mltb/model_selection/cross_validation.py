import warnings
warnings.filterwarnings('ignore')
import pandas as pd

def run_cv(X, y, Xtest, cv_generator, mdl, metric, task=None, verbose=False, calc_score=True):
    if isinstance(Xtest, (pd.DataFrame)):
        if Xtest.empty:
            Xtest = None

    kf = cv_generator
    kfolds = cv_generator.n_splits
    kf_name = kf.__class__.__name__
    kfy = y if kf_name == 'StratifiedKFold' else None

    oof = pd.Series()
    ytest = pd.DataFrame()
    if verbose: print(f'cv - fold', end='')
    for ix, (train_index, val_index) in enumerate(kf.split(X, kfy), 1):
        if verbose:
            print(f' {ix}', end='' if ix<kfolds else '\n')

        Xtrain = X.iloc[train_index]
        ytrain = y.iloc[train_index]
        Xval = X.iloc[val_index]

        # train
        mdl_cv = mdl.fit(Xtrain,ytrain)

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


def run_cv1(X, y, Xtest, mdl, metric, params, kfolds=5, task=None, kfold_seed=888, shuffle=False, verbose=False,
           calc_score=True):
    if not shuffle:
        kfold_seed = None
    if isinstance(Xtest, (pd.DataFrame)):
        if Xtest.empty:
            Xtest = None

    oof = pd.Series()
    ytest = pd.DataFrame()
    kf = StratifiedKFold(n_splits=kfolds, random_state=kfold_seed)
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

