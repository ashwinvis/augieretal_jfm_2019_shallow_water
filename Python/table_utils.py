from base import *
from paths import *
import scipy
import pandas as pd
from pandas import DataFrame


def float_format(val, *args):
    # if isinstance(val, int):
    if val.is_integer():
        return '{:d}'.format(int(val), *args)
    else:
        return '{:.3g}'.format(val, *args)

pd.options.display.float_format = float_format

def get_row(df, col_name, col_val):
    return df[df[col_name] == col_val]


def clean_up(df):
    df = df[
        df["$n$"]>=960
    ]
    df = df[df["$t_{stat}$"] < 0.8 * df["$t_{\max}$"]]
    return df

def sort_reindex(df, prefix=None):
    df = df.sort_values(by=[r'$c$', r'$n$', r'$Bu$', EFR])
    if prefix is not None:
        df.index = ['{}{}'.format(prefix, i) for i in range(1,len(df) + 1)]
    else:
        df.index = range(1,len(df) + 1)
    # df = df.reindex(range(1,len(df) + 1))
    return df


from IPython import display

def to_latex(df, strip_cols, filename=None, **kwargs):
    df = df.drop(strip_cols, axis=1)

    output = df.to_latex(escape=False, **kwargs)

    if filename == '':
        print(output)
    elif filename == 'jupyter':
        display.display_html(df)
    else:
        with open(filename, 'a') as f:
            f.write(output)


print('scipy ==', scipy.__version__, '; pandas ==', pd.__version__, '; numpy ==', np.__version__)  # important!
