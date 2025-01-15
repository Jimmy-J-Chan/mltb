from sklearn.metrics import root_mean_squared_error as rmse
import pandas as pd
import numpy as np


def climb_hill(oofs, y, ypreds, obj, metric, neg_w=False, step=0.01, return_oof_ens=False):

    STOP = False
    scores = {}
    i = 0

    oof_df = oofs
    test_preds = ypreds

    # Compute CV scores on the train data
    for col in oof_df.columns:
        scores[col] = metric(y, oof_df[col])

    # Sort CV scores
    if obj == "minimize":
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=False)}
    elif obj == "maximize":
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    else:
        raise ValueError("Please provide valid hillclimbing objective (minimize or maximize)")

    print(f"Models to be ensembled | ({len(scores)} total)\n")
    max_model_len = max(len(model) for model in scores.keys())

    # Display models with their associated metric score
    for e, (model, score) in enumerate(scores.items()):
        model_padding = " " * (max_model_len - len(model))
        score_str = f"{score:.5f}".rjust(2)
        if e == 0:
            print(f"{model}:{model_padding} {score_str} (best solo model)")
        else:
            print(f"{model}:{model_padding} {score_str}")
    print()

    oof_df = oof_df[list(scores.keys())]
    test_preds = test_preds[list(scores.keys())]
    current_best_ensemble = oof_df.iloc[:, 0]
    current_best_test_preds = test_preds.iloc[:, 0]
    MODELS = oof_df.iloc[:, 1:].copy()
    history = [metric(y, current_best_ensemble)]

    if step > 0:
        if neg_w:
            weight_range = np.arange(-0.5, 0.51, step)
        else:
            weight_range = np.arange(step, 0.51, step)
    else:
        raise ValueError("step size must be a positive number")

    decimal_length = len(str(step).split(".")[1])
    eval_metric_name = metric.__name__

    print(f"[Data preparation completed successfully] - [Initiate hill climbing] \n")

    # Hill climbing
    w_ens_hist = pd.DataFrame(columns=MODELS.columns)
    w_ens_hist.loc[0] = 0.
    w_ens_hist.loc[0, list(scores.keys())[0]] = 1.
    while not STOP:

        i += 1
        potential_new_best_cv_score = metric(y, current_best_ensemble)
        k_best, wgt_best = None, None

        for k in MODELS:
            for wgt in weight_range:
                potential_ensemble = (1 - wgt) * current_best_ensemble + wgt * MODELS[k]
                cv_score = metric(y, potential_ensemble)

                if obj == "minimize":
                    if cv_score < potential_new_best_cv_score:
                        potential_new_best_cv_score = cv_score
                        k_best, wgt_best = k, wgt

                elif obj == "maximize":
                    if cv_score > potential_new_best_cv_score:
                        potential_new_best_cv_score = cv_score
                        k_best, wgt_best = k, wgt

        if k_best is not None:
            current_best_ensemble = (1 - wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
            current_best_test_preds = (1 - wgt_best) * current_best_test_preds + wgt_best * test_preds[k_best]
            MODELS.drop(k_best, axis=1, inplace=True)

            # calc ensemble effective weights
            current_mdls = w_ens_hist.loc[i-1][w_ens_hist.loc[i-1]>0].index
            w_ens_hist.loc[i] = w_ens_hist.loc[i - 1]
            w_ens_hist.loc[i, current_mdls] = w_ens_hist.loc[i, current_mdls] * (1-wgt_best)
            w_ens_hist.loc[i, k_best] = wgt_best

            if MODELS.shape[1] == 0:
                STOP = True

            if wgt_best > 0:
                print(
                    f'Iteration: {i} | Model added: {k_best} | Best weight: {wgt_best:.{decimal_length}f} | Best {eval_metric_name}: {potential_new_best_cv_score:.5f}')
            elif wgt_best < 0:
                print(
                    f'Iteration: {i} | Model added: {k_best} | Best weight: {wgt_best:.{decimal_length}f} | Best {eval_metric_name}: {potential_new_best_cv_score:.5f}')
            else:
                print(
                    f'Iteration: {i} | Model added: {k_best} | Best weight: {wgt_best:.{decimal_length}f} | Best {eval_metric_name}: {potential_new_best_cv_score:.5f}')

            history.append(potential_new_best_cv_score)

        else:
            STOP = True

    w_ens = w_ens_hist.iloc[-1].loc[oofs.columns]
    if return_oof_ens:
        return current_best_test_preds, w_ens, current_best_ensemble
    else:
        return current_best_test_preds, w_ens


def hill_climbers_ens(oofs, y, ypreds, obj=None, metric=None, neg_w=False, step=0.01, return_oof_ens=False):
    """
    get ensemble weights using hill climbers
    https://github.com/Matt-OP/hillclimbers
    """
    ypreds_ens, w_ens = climb_hill(oofs, y, ypreds, obj, metric, neg_w=False, step=step, return_oof_ens=False)
    return ypreds_ens, w


if __name__ == '__main__':
    # load preds
    import pandas as pd
    preds = pd.read_pickle(r"C:\Users\User\PycharmProjects\mltb\mltb\data\preds.pkl")
    oofs = preds['oofs']
    oofs.columns = oofs.columns.astype(str)
    ytrain = preds['ytrain']
    ypreds = preds['ypreds']
    ypreds.columns = ypreds.columns.astype(str)

    ypreds_ens, w = hill_climbers_ens(oofs, ytrain, ypreds, 'minimize', rmse)
    pass