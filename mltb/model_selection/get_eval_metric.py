

def get_eval_metric(cv_task=None):
    eval_metrics = {'regression': 'rmse',
                    'classification': 'auc',
                    'binary_class': 'binary_logloss',
                    }
    if cv_task not in eval_metrics.keys():
        raise AttributeError(f"Could not map eval metric for task: {cv_task}")
    return eval_metrics[cv_task]