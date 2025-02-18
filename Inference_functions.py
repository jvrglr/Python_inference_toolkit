def learn_dif(xs,ts,th=1,N=100):
    """
    Learn diffusion function for time series {ts,xs} using quadratic variation estimator.
    The function assumes constant binning (grid) in temporal data.
    N-> number of bins
    th-> minimum number of counts per bin to do calculation
    """
    dt = ts[1]-ts[0]
    xmax = max(xs)
    xmin = min(xs)
    dx = (xmax-xmin)/N
    Dn = np.zeros(N+1)
    count = np.zeros(N+1)
    for ww,x in enumerate(xs[:-1:]):
        n = int((x-xmin)/dx)
        Dn[n] += (xs[ww+1]-x)**2.0
        count[n] += 1
    Dn = sqrt(Dn/count/dt)
    x_bins = np.arange(N+1)*dx+xmin
    threshold = th
    mask = count >= threshold
    count = count[mask]
    Dn = Dn[mask]
    x_bins = x_bins[mask]
    return x_bins,Dn,count

def learn_drift(xs,ts,th=1,N=100):
    """
    Learn drift function for time series {ts,xs} using firt moment of the jump
    The function assumes constant binning (grid) in temporal data.
    N-> number of bins
    th-> minimum number of counts per bin to do calculation
    """
    dt = ts[1]-ts[0]
    xmax = max(xs)
    xmin = min(xs)
    dx = (xmax-xmin)/N
    Fn = np.zeros(N+1)
    count = np.zeros(N+1)
    for ww,x in enumerate(xs[:-1:]):
        n = int((x-xmin)/dx)
        Fn[n] += (xs[ww+1]-x)
        count[n] += 1
    Fn = Fn/count/dt
    x_bins = np.arange(N+1)*dx+xmin
    threshold = th
    mask = count >= threshold
    count = count[mask]
    Fn = Fn[mask]
    x_bins = x_bins[mask]
    return x_bins,Fn,count
