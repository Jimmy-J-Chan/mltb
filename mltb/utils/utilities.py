import pandas as pd
import time


def cc(*dfs, columns=[], axis=1):
    dfc = pd.concat(dfs, axis=axis)
    if len(dfc.columns)==len(columns):
        dfc.columns = columns
    return dfc


def tic(prefix='', tictoc_var='tictoc_start_time', print_out=False, end=None):
    globals()[tictoc_var] = time.time()

    if print_out:
        prefix = prefix + 'Started at ' + time.strftime('%d %b, %H:%M') + ' '

    print(prefix, end=end)

def toc(prefix='', tictoc_var='tictoc_start_time', secs_elapsed=False, print_out=True, print_time=False, end='\n'):
    du = time.time() - globals()[tictoc_var]
    h = round(du//3600)
    m = round((du % 3600)//60)

    if secs_elapsed:
        return du

    if m>0:
        s = round((du & 3600) & 60)
    else:
        s = round((du % 3600) % 60, 2)

    if print_time & (h>24):
        suffix = f" at {time.strftime('%d %b, %H:%M')}"
    elif print_time:
        suffix = f" at {time.strftime('%H:%M')}"
    else:
        suffix = ''

    if len(prefix)>0:
        prefix = prefix + ' - '

    if du < 60:
        elapse = f"{s} sec"
    elif du < 3600:
        elapse = f"{m}m:{s}s"
    else:
        elapse = f"Elapse time: {h}h:{m}m:{s}s"

    if print_out:
        print(prefix + elapse + suffix, end=end)
    else:
        return prefix + elapse + suffix



if __name__ == '__main__':
    fn = r"C:\Users\Jimmy\PycharmProjects\health_insr_cross_sell\data\train_original.csv"
    df = pd.read_csv(fn)

    df1 = df.iloc[:100]
    df2 = df.iloc[-100:]
    dfc = cc(df1, df2, columns=['df1','df2'])
    pass