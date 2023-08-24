import numpy as np
import pandas as pd
from utils.fast_ewma import _ewma

def get_tick_bar(ticks : pd.DataFrame, freq : int) -> pd.DataFrame:
    """
    Get tick bars

    Parameters
    ----------
    ticks : pd.DataFrame
        DataFrame of ticks
    freq : int
        Frequency of tick bars

    Returns
    -------
    np.ndarray
        Array of tick bars
    """
    tick_bars = np.zeros((len(range(0, len(ticks)-freq, freq)), 6),dtype=object)
    for i in range(0, len(ticks)-freq, freq):
        tick_bars[int(i/freq),0] = ticks.loc[i+freq-1,'time']
        tick_bars[int(i/freq),1] = ticks.loc[i,'price']
        tick_bars[int(i/freq),2] = ticks.loc[i:i+freq,'price'].max()
        tick_bars[int(i/freq),3] = ticks.loc[i:i+freq,'price'].min()
        tick_bars[int(i/freq),4] = ticks.loc[i+freq-1,'price']
        tick_bars[int(i/freq),5] = ticks.loc[i:i+freq,'volume'].sum()

    # convert to pandas dataframe and name columns as time open high low close volume
    tick_bars = pd.DataFrame(tick_bars, columns=['time','open','high','low','close','volume'])
    return tick_bars

def get_volume_bar(ticks : pd.DataFrame, vol : int) -> pd.DataFrame:
    """
    Get volume bars

    Parameters
    ----------
    ticks : pd.DataFrame
        DataFrame of ticks
    freq : int
        Frequency of volume bars

    Returns
    -------
    np.ndarray
        Array of volume bars
    """
    volume_bars = np.zeros((len(ticks), 6),dtype=object)
    last_tick = 0
    curr_vol = 0
    it = 0
    for i in range(len(ticks)):
        curr_vol += ticks.loc[i,'volume']
        if curr_vol >= vol:
            volume_bars[it,0] = ticks.loc[last_tick,'time']
            volume_bars[it,1] = ticks.loc[last_tick,'price']
            volume_bars[it,2] = ticks.loc[last_tick:i,'price'].max()
            volume_bars[it,3] = ticks.loc[last_tick:i,'price'].min()
            volume_bars[it,4] = ticks.loc[i,'price']
            volume_bars[it,5] = ticks.loc[last_tick:i,'volume'].sum()
            last_tick = i+1
            curr_vol = 0
            it += 1
    volume_bars = volume_bars[:it,:]
    # convert to pandas dataframe and name columns as time open high low close volume
    volume_bars = pd.DataFrame(volume_bars, columns=['time','open','high','low','close','volume'])
    return volume_bars

def get_dollar_bar(ticks : pd.DataFrame, dol : int) -> pd.DataFrame:
    """
    Get dollar bars

    Parameters
    ----------
    ticks : pd.DataFrame
        DataFrame of ticks
    freq : int
        Frequency of dollar bars

    Returns
    -------
    np.ndarray
        Array of dollar bars
    """
    dollar_bars = np.zeros((len(ticks), 6),dtype=object)
    last_tick = 0
    curr_dol = 0
    it = 0
    for i in range(len(ticks)):
        curr_dol += ticks.loc[i,'dollar']
        if curr_dol >= dol:
            dollar_bars[it,0] = ticks.loc[last_tick,'time']
            dollar_bars[it,1] = ticks.loc[last_tick,'price']
            dollar_bars[it,2] = ticks.loc[last_tick:i,'price'].max()
            dollar_bars[it,3] = ticks.loc[last_tick:i,'price'].min()
            dollar_bars[it,4] = ticks.loc[i,'price']
            dollar_bars[it,5] = ticks.loc[last_tick:i,'volume'].sum()
            last_tick = i+1
            curr_dol = 0
            it += 1
    dollar_bars = dollar_bars[:it,:]
    # convert to pandas dataframe and name columns as time open high low close volume
    dollar_bars = pd.DataFrame(dollar_bars, columns=['time','open','high','low','close','volume'])
    return dollar_bars

def get_imbalance_dollar_bar(ticks : pd.DataFrame, window : int) -> pd.DataFrame:
    """
    Get dollar bars

    Parameters
    ----------
    ticks : pd.DataFrame
        DataFrame of ticks
    window : int
        Window for exponential moving average
    Returns
    -------
    pd.DataFrame
        DataFrame of dollar bars
    """
    dollar_bars = np.zeros((len(ticks), 6),dtype=object)
    last_tick = 0
    curr_imb = 0
    j = 0
    # init b,t,d to ndarray
    b,t,d = np.zeros(len(ticks),dtype=np.float64),np.zeros(len(ticks),dtype=np.float64),ticks['dollar'].values.astype(np.float64)
    b[0] = 1
    t[0] = 10000
    for i in range(1,len(ticks)):
        # determine b
        if ticks.loc[i,'price'] == ticks.loc[i-1,'price']:
            b[i] = b[i-1]
        elif ticks.loc[i,'price'] > ticks.loc[i-1,'price']:
            b[i] = 1
        else:
            b[i] = -1
        # calculate dollar imbalance
        curr_imb += b[i]*d[i]
        # calculate expected dollar imbalance
        expected_imb = _ewma(t,window,j+1)*_ewma(b,window,i+1)*_ewma(d,window,i+1)
        # check if dollar imbalance is greater than expected
        if abs(curr_imb) >= abs(expected_imb):
            dollar_bars[j,0] = ticks.loc[i,'time']
            dollar_bars[j,1] = ticks.loc[last_tick,'price']
            dollar_bars[j,2] = ticks.loc[last_tick:i,'price'].max()
            dollar_bars[j,3] = ticks.loc[last_tick:i,'price'].min()
            dollar_bars[j,4] = ticks.loc[i,'price']
            dollar_bars[j,5] = ticks.loc[last_tick:i,'volume'].sum()
            j += 1
            t[j] = i+1-last_tick
            last_tick = i+1
            curr_imb = 0
            
    dollar_bars = dollar_bars[:j,:]
    # convert to pandas dataframe and name columns as time open high low close volume
    dollar_bars = pd.DataFrame(dollar_bars, columns=['time','open','high','low','close','volume'])
    return dollar_bars

def get_run_dollar_bar(ticks : pd.Series, window : int, indices_only = False) -> pd.DataFrame:
    """
    Get dollar bars

    Parameters
    ----------
    ticks : pd.DataFrame
        DataFrame of ticks
    window : int
        Window for exponential moving average

    Returns
    -------
    pd.DataFrame
        Array of dollar bars
    or
    np.ndarray
        Array of indices of dollar bars
        not in binary format !
    """
    if indices_only:
        indices = np.zeros(len(ticks),dtype=np.int64)
    else:
        run_bars = np.zeros((len(ticks), 7),dtype=object)
    b = np.zeros(len(ticks),dtype=np.float64)
    b[0] = 1
    vp,vn = np.zeros(len(ticks),dtype=np.float64),np.zeros(len(ticks),dtype=np.float64)
    vpi,vni = 0,0
    t = np.zeros(len(ticks),dtype=np.float64)
    j = 0
    t[0] = 10000
    p = np.zeros(len(ticks),dtype=np.float64)
    run,sp,sn,success = 0,0,0,0
    last_tick = 0
    for i in range(1,len(ticks)):
        # update variables
        if ticks.loc[i,'price'] == ticks.loc[i-1,'price']:
            b[i] = b[i-1]
            if b[i] == 1:
                vp[vpi] = ticks.loc[i,'dollar']
                vpi += 1
                success += 1
                sp += vp[vpi-1]
            else:
                vn[vni] = ticks.loc[i,'dollar']
                vni += 1
                sn -= vn[vni-1]
            p[i-1] = success/i
        elif ticks.loc[i,'price'] > ticks.loc[i-1,'price']:
            b[i] = 1
            vp[vpi] = ticks.loc[i,'dollar']
            vpi += 1
            success += 1
            p[i-1] = success/i
            sp += vp[vpi-1]
        else:
            b[i] = -1
            vn[vni] = ticks.loc[i,'dollar']
            vni += 1
            p[i-1] = success/i
            sn -= vn[vni-1]
        run = max(sp,-sn)
        # calculate expected run
        expected_run = _ewma(t,window,j+1)*max(_ewma(p,window,i)*_ewma(vp,window,vpi),(1-_ewma(p,window,i))*_ewma(vn,window,vni))

        if run >= expected_run:
            if indices_only:
                indices[j] = ticks.at[i,'id']
            else:
                run_bars[j,0] = ticks.loc[i,'time']
                run_bars[j,1] = ticks.loc[last_tick,'price']
                run_bars[j,2] = ticks.loc[last_tick:i,'price'].max()
                run_bars[j,3] = ticks.loc[last_tick:i,'price'].min()
                run_bars[j,4] = ticks.loc[i,'price']
                run_bars[j,5] = ticks.loc[last_tick:i,'volume'].sum()
                run_bars[j,6] = ticks.loc[i,'id']
            t[j+1],j = i+1-last_tick,j+1
            last_tick = i+1
            sp,sn,run = 0,0,0
            
    if indices_only:
        return indices[:j]
    # truncate arrays
    run_bars = run_bars[:j,:]
    # convert to pandas dataframe and name columns as time open high low close volume
    run_bars = pd.DataFrame(run_bars, columns=['time','open','high','low','close','volume','id'])
    return run_bars

def get_cusum_events(ticks : pd.Series, h : int) -> np.ndarray:
    """
    Get t events

    Parameters
    ----------
    ticks : pd.Series
        DataFrame of ticks
    h : int
        Threshold for t events

    Returns
    -------
    np.ndarray
        Array of t events
    """
    t_events,sp,sn = np.zeros(len(ticks),dtype=np.bool),0,0
    for i in range(1,len(ticks)):
        sp , sn = max(0,sp+ticks.iloc[i]-ticks.iloc[i-1]),min(0,sn+ticks.iloc[i]-ticks.iloc[i-1])
        if sp >= h:
            t_events[i] = True
            sp = 0
        elif sn <= -h:
            t_events[i] = True
            sn = 0
    print("Number of samples percentage: ",sum(t_events)/len(ticks))
    return t_events

