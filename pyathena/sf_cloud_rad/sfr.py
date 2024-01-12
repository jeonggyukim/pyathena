from scipy.optimize import curve_fit

def get_SFR_mean(h, t0_SF=0.0, t1_SF=90.0):
    """
    Perform least-square fitting to SF history (from t_* to t1_SF)

    Returns
    -------
    Slope (mean SFR) in units of Msun/Myr
    """

    def func(x, a):
        return a*x

    Mstar_final = max(h['Mstar'].values)

    if t0_SF == 0.0:
        idx_SF0, = h['Mstar'].to_numpy().nonzero()
        t0 = h['time'][idx_SF0[0]-1]
    else:
        t0 = h['time'][h.Mstar > 1e-2*t0_SF*Mstar_final].values[0]

    t1 = h['time'][h.Mstar > 1e-2*t1_SF*Mstar_final].values[0]

    h_ = h[(h['time'] >= t0) & (h['time'] <= t1)]
    xdata = h_['time'].values - t0
    ydata = h_['Mstar'].values
    SFR_mean, pcov = curve_fit(func, xdata, ydata)

    #plt.plot(h['time'],h['Mstar'])
    # plt.plot(h_['time'] - rr['t_*'],h_['Mstar'],ls='--')
    # plt.plot(xdata, func(xdata, *popt))

    res = dict()
    res['t0'] = t0
    res['t1'] = t1
    res['SFR_mean'] = SFR_mean[0]
    res['h'] = h_

    return res
