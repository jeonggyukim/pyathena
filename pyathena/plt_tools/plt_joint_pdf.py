import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def plt_joint_pdf(x, y, hexbin_args=dict(), bins=30,
                  weights=None, gs=None):
    """Plot a 2D joint PDF with marginalized 1D histograms.

    Creates a figure with a central hexbin 2D histogram and marginal
    histograms along the x and y axes.

    Parameters
    ----------
    x : array-like
        Data for the x-axis.
    y : array-like
        Data for the y-axis.
    hexbin_args : dict, optional
        Arguments passed to :func:`matplotlib.axes.Axes.hexbin`. Must include
        an ``extent`` key (``[xmin, xmax, ymin, ymax]`` in log10 for log
        scales) for the marginal histograms to align correctly.
        Default sets ``mincnt=1``, ``xscale='log'``, ``yscale='log'``.
    bins : int, optional
        Number of bins for the marginal histograms. Default is 30.
    weights : array-like, optional
        Weights applied to both the hexbin and marginal histograms.
        Default is ``None``.
    gs : matplotlib.gridspec.GridSpec, optional
        Existing GridSpec to draw into. If ``None``, a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Central 2D hexbin axes.
    axx : matplotlib.axes.Axes
        Top marginal histogram axes (x-axis projection).
    axy : matplotlib.axes.Axes
        Right marginal histogram axes (y-axis projection).
    """

    if gs is None:
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(4, 4)

    else:
        fig = plt.gcf()

    ax = fig.add_subplot(gs[1:4,0:3])
    axx = fig.add_subplot(gs[0,0:3])
    axy = fig.add_subplot(gs[1:4,3])

    _hexbin_args = dict()
    _hexbin_args['mincnt'] = 1
    _hexbin_args['xscale'] = 'log'
    _hexbin_args['yscale'] = 'log'
    _hexbin_args.update(**hexbin_args)

    if weights is not None:
        _hexbin_args['C'] = weights
        _hexbin_args['reduce_C_function'] = np.sum

    # joint pdfs
    ax.hexbin(x, y, **_hexbin_args)

    # plot marginalized pdfs
    if _hexbin_args['xscale'] == 'log':
        h, bine = np.histogram(np.log10(x),weights=weights, bins=bins, density=True)
        axx.step(10.0**bine[1:], h, 'k-')
        axx.set_xscale('log')
        axx.set_xlim(10**_hexbin_args['extent'][0], 10**_hexbin_args['extent'][1])
    else:
        h, bine = np.histogram(x, weights=weights, bins=bins, density=True)
        axx.step(bine[1:], h, 'k-')
        axx.set_xlim(_hexbin_args['extent'][0], _hexbin_args['extent'][1])

    if _hexbin_args['yscale'] == 'log':
        h, bine = np.histogram(np.log10(y), weights=weights, bins=bins, density=True)
        axy.step(h, 10.0**bine[1:], 'k-')
        axy.set_yscale('log')
        axy.set_ylim(10**_hexbin_args['extent'][2], 10**_hexbin_args['extent'][3])
    else:
        h, bine = np.histogram(y, weights=weights, bins=bins, density=True)
        axy.step(h, bine[1:], 'k-')
        axy.set_ylim(_hexbin_args['extent'][2], _hexbin_args['extent'][3])

    # Turn off tick labels on marginals
    plt.setp(axx.get_xticklabels(), visible=False)
    plt.setp(axy.get_yticklabels(), visible=False)

    # Set labels on marginals
    axx.set_ylabel('pdf')
    axy.set_xlabel('pdf')

    return ax, axx, axy
