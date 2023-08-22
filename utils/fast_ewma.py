import numpy as np
from numba import jit
from numba import float64
from numba import int64

@jit((float64[:], int64, int64), nopython=True, nogil=True)
def _ewma(arr : np.ndarray, window:int64, end:int64) -> float64:
    r"""Exponentialy weighted moving average specified by a decay ``window``
    to provide better adjustments for small windows via:
        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).
        Optimized denominator : ((1-a)^n - 1) / (1-a - 1)
    Parameters
    ----------
    arr : np.ndarray, float64
        A single dimenisional numpy array
    window : int64
        The decay window, or 'span'
    end : int64
        The index to end the ewma calculation at (exclusive)
    Returns
    -------
    float64
        ewma at index `end-1`
    """
    if end > arr.shape[0]:
        raise IndexError('end parameter is out of bounds')
    if end == 0:
        return 0
    # adjust window
    window = min(window, end)
    alpha = 2 / float(window + 1)
    nominator = arr[end-window]
    for i in range(end-window+1, end):
        nominator = nominator*(1-alpha) + arr[i]
    return nominator/ (((1-alpha)**window - 1) / (-alpha))