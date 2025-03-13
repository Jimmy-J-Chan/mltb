import pandas as pd
from mltb.struct import Struct


def fc_lgbm_base():
    from lightgbm import LGBMRegressor
    from mltb.model_selection.cross_validation import run_cv
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.model_selection import StratifiedKFold

    # read in train data
    train = pd.read_parquet(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\train.parquet")
    Xtest = pd.read_parquet(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\test.parquet").drop(columns=['id'])
    X = train.drop(columns=['id','price'])
    y = train['price']

    # get preds
    cv_generator = StratifiedKFold(n_splits=5)
    ypreds = pd.DataFrame()
    oofs = pd.DataFrame()
    mdl = LGBMRegressor(verbose=-1)
    mdl2 = LGBMRegressor(learning_rate=0.01, num_leaves=100, n_estimators=1000, colsample_bytree=0.7, verbose=-1)
    mdl3 = LGBMRegressor(learning_rate=0.05, num_leaves=50, n_estimators=500, colsample_bytree=0.9, verbose=-1)
    mdl4 = LGBMRegressor(learning_rate=0.2, num_leaves=50, n_estimators=100, colsample_bytree=0.9, verbose=-1)

    # get more mdls
    mdlnest = [LGBMRegressor(verbose=-1, n_estimators=c) for c in range(40, 200, 20)]
    mdllr = [LGBMRegressor(verbose=-1, learning_rate=c) for c in [0.0001,0.0005,0.001,.005,0.01,0.02,0.03,0.04,0.05]]

    for ix, m in enumerate([mdl, mdl2, mdl3, mdl4]+mdlnest+mdllr, 1):
        tmp_oof, score, ytest = run_cv(X, y, Xtest, cv_generator, m, rmse, task='regression')
        print(f'{ix}/21 - score: {score}')
        oofs = pd.concat([oofs, tmp_oof.to_frame(int(score))], axis=1)
        ypreds = pd.concat([ypreds, ytest.to_frame(int(score))], axis=1)

    # save ypreds and oofs
    preds = Struct({'ytrain':y,'oofs':oofs, 'ypreds':ypreds})
    preds.to_pickle(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\ypreds_lgbm.pkl")
    pass

def fix_dup_cols():
    # load preds
    preds = pd.read_pickle(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\ypreds_lgbm.pkl")
    y = preds['ytrain']
    oofs = preds['oofs']
    ypreds = preds['ypreds']

    # fix dupe cols
    oofs.columns = [73923]+ list(oofs.columns[1:])
    ypreds.columns = oofs.columns

    preds = Struct({'ytrain':y,'oofs':oofs, 'ypreds':ypreds})
    preds.to_pickle(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\ypreds_lgbm.pkl")
    pass

if __name__ == '__main__':
    # fc_lgbm_base()
    fix_dup_cols()
    pass