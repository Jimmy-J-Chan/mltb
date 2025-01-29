import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def diversity_selection(lb=None):

    feats = [c for c in lb.columns if c!='score']
    lb_calc = lb.copy()

    # encode
    bool_cols = [c for c in lb_calc.columns if lb_calc[c].dtype=='bool']
    lb_calc.loc[:,bool_cols] = lb_calc.loc[:,bool_cols].astype(int)
    cat_cols = [c for c in lb_calc.columns if (lb_calc[c].dtype=='object')|(lb_calc[c].dtype=='category')]
    for cat_col in cat_cols:
        cmap = {c:ix for ix, c in enumerate(lb_calc[cat_col].unique())}
        lb_calc.loc[:,cat_col] = lb_calc.loc[:,cat_col].map(cmap)

    lb_calc = lb_calc[feats].head(int(0.33*len(lb))) # pick top 33%
    sim = pd.DataFrame(cosine_similarity(lb_calc), index=lb_calc.index, columns=lb_calc.index)

    # iteratively pick trials that are diverse in hyperparamters as measured using cosine similarity
    best_trial = lb_calc.index[0]
    trials2keep = [best_trial]
    threshold = 0.7
    for trial in sim.columns:
        tmp = sim.loc[trials2keep, trial] # sim scores between trials already picked against potential trial to be added
        if (tmp<=threshold).all(): # if all trials picked < threshold against trial to be added, add to trials to keep
            trials2keep.append(trial)

    return trials2keep
