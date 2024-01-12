from scipy.optimize import curve_fit

def get_SFR_mean(df, h, t0_SF=0.0, t1_SF=90.0):
    """
    Perform linear least-squares fitting to SF history (from t0 to t1)

    Parameters
    ----------
    df : pandas dataframe
       summary returned by s.get_summary()
    h : pandas dataframe
       History
    t0_SF, t1_SF : floats between 0 and 100
       Set time interval (e.g., for 10 and 90, consider t_10%,* < t < t_90%,*)

    Returns
    -------
    Slope (mean SFR) in units of Msun/Myr
    """

    def func(x, a, b):
        return a*x + b

    Mstar_final = max(h['M_sp'].values)

    if t0_SF == 0.0:
        idx_SF0, = h['M_sp'].to_numpy().nonzero()
        t0 = h['time'][idx_SF0[0]-1]
    else:
        t0 = h['time'][h['M_sp'] > 1e-2*t0_SF*Mstar_final].values[0]

    t1 = h['time'][h['M_sp'] > 1e-2*t1_SF*Mstar_final].values[0]
    print(t0,t1)
    M_sp_t0 = h.loc[h['time'] == t0, 'M_sp'].values
    M_sp_t1 = h.loc[h['time'] == t1, 'M_sp'].values

    h_ = h[(h['time'] >= t0) & (h['time'] <= t1)]
    xdata = h_['time'].values - t0
    ydata = h_['M_sp'].values
    popt, pcov = curve_fit(func, xdata, ydata)

    #plt.plot(h['time'],h['Mstar'])
    # plt.plot(h_['time'] - rr['t_*'],h_['Mstar'],ls='--')
    # plt.plot(xdata, func(xdata, *popt))

    res = dict()
    res['t0'] = t0
    res['t1'] = t1
    res['SFR_mean'] = popt[0]
    res['popt'] = popt
    #res['h'] = h_
    res['M_sp_t0'] = M_sp_t0
    res['M_sp_t1'] = M_sp_t1

    res['tff'] = df['tff']
    res['tdyn'] = df['tdyn']
    res['alpha_vir'] = df['alpha_vir']
    res['eps_ff'] = df['tff']/(df['M']/popt[0])

    return res
