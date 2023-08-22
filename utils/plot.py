import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_kde(df):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    col_index = 0
    for i in range(2):
        for j in range(3):
            if col_index < len(df.columns):
                df.plot(kind='kde', ax=axs[i, j], y=df.columns[col_index], title=df.columns[col_index])
                col_index += 1
                axs[i,j].set_xlim((-3,3))
    plt.tight_layout()
    plt.show()