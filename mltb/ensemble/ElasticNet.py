from sklearn.linear_model import ElasticNet
from mltb.model_selection.cross_validation import run_cv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error as rmse

def ElasticNet_ensemble(oofs, y, ypreds, mdl_params):
    """
    Linear regression with combined L1 and L2 priors as regularizer.
    *** ONLY FOR REGRESSION PROBLEMS ***
    """
    Xtest = ypreds
    cv_generator = StratifiedKFold(n_splits=5)
    mdl = ElasticNet(positive=True, fit_intercept=False, **mdl_params)
    oof, score, ypreds_ens, mdls = run_cv(oofs, y, Xtest, cv_generator, mdl, rmse, task='regression', return_mdls=True)

    w = pd.concat([pd.Series(mdls[n].coef_) for n in range(len(mdls))], axis=1).mean(axis=1)
    w.index = Xtest.columns
    # ypreds_ens = (Xtest * avg_w).sum(axis=1)
    return ypreds_ens, w


if __name__ == '__main__':
    # load preds
    import pandas as pd
    preds = pd.read_pickle(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\ypreds_lgbm.pkl")
    oofs = preds['oofs']
    ytrain = preds['ytrain']
    ypreds = preds['ypreds']
    mdl_params = {}
    ypreds_ens, w = ElasticNet_ensemble(oofs, ytrain, ypreds, mdl_params)
    pass