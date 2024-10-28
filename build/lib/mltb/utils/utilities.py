import pandas as pd


def cc(*dfs, columns=[], axis=1):
    dfc = pd.concat(dfs, axis=axis)
    if len(dfc.columns)==len(columns):
        dfc.columns = columns
    return dfc



if __name__ == '__main__':
    fn = r"C:\Users\Jimmy\PycharmProjects\health_insr_cross_sell\data\train_original.csv"
    df = pd.read_csv(fn)

    df1 = df.iloc[:100]
    df2 = df.iloc[-100:]
    dfc = cc(df1, df2, columns=['df1','df2'])
    pass