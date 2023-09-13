from fracdiff.sklearn import Fracdiff, FracdiffStat
import pandas as pd

def get_frac_diff_df(df,cols,d=None):
    if d is None:
        fd = FracdiffStat()
        fracdiff_df = pd.DataFrame(fd.fit_transform(df[cols]),index=df.index,columns=cols)
        return fracdiff_df,fd.d_
    else:
        fracdiff_df = pd.DataFrame(index=df.index)
        for i,col in enumerate(cols):
            fd = Fracdiff(d[i])
            fracdiff_df[col] = fd.fit_transform(df[col].values.reshape(-1, 1)).flatten()
        return fracdiff_df,d
    