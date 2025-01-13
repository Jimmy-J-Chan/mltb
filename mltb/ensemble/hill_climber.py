from sklearn.metrics import root_mean_squared_error as rmse
import pandas as pd
import numpy as np



def climb_hill(y, oof_pred_df, test_pred_df, objective, eval_metric, negative_weights=False, precision=0.01, return_oof_preds=False):

    STOP = False
    scores = {}
    i = 0

    oof_df = oof_pred_df
    test_preds = test_pred_df

    # Compute CV scores on the train data
    for col in oof_df.columns:
        scores[col] = eval_metric(y, oof_df[col])

    # Sort CV scores
    if objective == "minimize":
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=False)}
    elif objective == "maximize":
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
    history = [eval_metric(y, current_best_ensemble)]

    if precision > 0:
        if negative_weights:
            weight_range = np.arange(-0.5, 0.51, precision)
        else:
            weight_range = np.arange(precision, 0.51, precision)
    else:
        raise ValueError("precision must be a positive number")

    decimal_length = len(str(precision).split(".")[1])
    eval_metric_name = eval_metric.__name__

    print(f"[Data preparation completed successfully] - [Initiate hill climbing] \n")

    # Hill climbing
    while not STOP:

        i += 1
        potential_new_best_cv_score = eval_metric(y, current_best_ensemble)
        k_best, wgt_best = None, None

        for k in MODELS:
            for wgt in weight_range:
                potential_ensemble = (1 - wgt) * current_best_ensemble + wgt * MODELS[k]
                cv_score = eval_metric(y, potential_ensemble)

                if objective == "minimize":
                    if cv_score < potential_new_best_cv_score:
                        potential_new_best_cv_score = cv_score
                        k_best, wgt_best = k, wgt

                elif objective == "maximize":
                    if cv_score > potential_new_best_cv_score:
                        potential_new_best_cv_score = cv_score
                        k_best, wgt_best = k, wgt

        if k_best is not None:
            current_best_ensemble = (1 - wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
            current_best_test_preds = (1 - wgt_best) * current_best_test_preds + wgt_best * test_preds[k_best]
            MODELS.drop(k_best, axis=1, inplace=True)

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

    if return_oof_preds:
        return current_best_test_preds.values, current_best_ensemble.values
    else:
        return current_best_test_preds.values


def min_optimise_ens(oofs, y, ypreds):
    """
    get ensemble weights using hill climbers
    https://github.com/Matt-OP/hillclimbers
    """
    ypreds_ens = climb_hill(y, oofs, ypreds, "minimize", rmse, negative_weights=False)
    return ypreds_ens


if __name__ == '__main__':
    # load preds
    import pandas as pd
    preds = pd.read_pickle(r"C:\Users\Jimmy\PycharmProjects\mltb\data\used_car_prices\ypreds_lgbm.pkl")
    oofs = preds['oofs']
    oofs.columns = oofs.columns.astype(str)
    ytrain = preds['ytrain']
    ypreds = preds['ypreds']
    ypreds.columns = ypreds.columns.astype(str)

    ypreds_ens = min_optimise_ens(oofs, ytrain, ypreds)
    pass