from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error as rmse
from scipy.optimize import minimize
import pandas as pd

def objective(w, oofs, y):
    ypred = (oofs*w).sum(axis=1)
    return rmse(y, ypred)


def min_optimise_ens(oofs, y, ypreds, method='SLSQP'):
    """
    get ensemble weights using scipy optimize using chosen method
    - ’Nelder-Mead’
    - ’SLSQP’
    """
    starting_values = [0]* oofs.shape[1]
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)}) # w sum to 1
    bounds = [(0,None)] * oofs.shape[1]  # w>0

    res = minimize(objective,
                   x0=starting_values,
                   args=(oofs, y),
                   method=method,
                   bounds = bounds,
                   constraints=cons,
                   tol=0.0003)

    w = pd.Series(res['x'], index=oofs.columns)
    ypreds_ens = (ypreds*w).sum(axis=1)

    return ypreds_ens, w


if __name__ == '__main__':
    # load preds
    import pandas as pd
    preds = pd.read_pickle(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\ypreds_lgbm.pkl")
    oofs = preds['oofs']
    ytrain = preds['ytrain']
    ypreds = preds['ypreds']
    ypreds_ens, w = min_optimise_ens(oofs, ytrain, ypreds, method='SLSQP')
    pass