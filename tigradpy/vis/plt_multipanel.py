#!/usr/bin/env python

from __future__ import print_function
import os,sys
import yt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

sys.path.insert(0, '../../')
import tigradpy as tp

def yt_multipanel(ds, kind='slice', axis='z',
                  fields=['nH', 'xn', 'G0prime0', 'G0prime1']):
    """
    Make 2x2 multi-panel plot using yt
    Based on
    http://yt-project.org/doc/cookbook/complex_plots.html#multipanel-with-axes-labels
    
    Parameters
    ----------
       ds : yt Dataset
       kind : string
           slice or projection
       axis : string
           x or y or z
       fields: list_like
           List of fields
    
    Returns
    -------
       p : yt.visualization.plot_window.AxisAlignedSlicePlot or
           yt.visualization.plot_window.ProjectionPlot
           yt plot object
    """
    
    if kind == 'slice':
        p = yt.SlicePlot(ds, axis, fields)
    elif kind == 'projection':
        p = yt.ProjectionPlot(ds, axis, fields, weight_field='cell_volume')

    nf = len(fields)
        
    if axis == 'x' or axis == 'y':
        nrows_ncols = (1,nf)
        figsize = (12,12)
        cbar_location = "right"
    else:
        nrows_ncols = (2,2)
        figsize = (12,12)
        cbar_location = "right"
    
    fig = plt.figure(figsize=figsize)
    grid = AxesGrid(fig, (0.075,0.075,0.85,0.85),
                    nrows_ncols=nrows_ncols, axes_pad=(1.2,0.1),
                    label_mode="1", share_all=True,
                    cbar_location=cbar_location, cbar_mode="each",
                    cbar_size="4%", cbar_pad="2%")

    p.set_zlim('nH', 1e-4, 1e2)
    p.set_zlim('xn', 0.0, 1.0)
    p.set_zlim('G0prime0', 1e-4, 1e2)
    p.set_zlim('G0prime1', 1e-2, 1e2)
    
    p.set_cmap(field=('athena','nH'), cmap='Spectral_r')
    p.set_cmap(field=('athena','xn'), cmap='viridis')
    p.set_cmap(field='G0prime0', cmap='plasma')
    p.set_cmap(field='G0prime1', cmap='plasma')

    for i, field in enumerate(fields):
        plot = p.plots[field]
        plot.figure = fig
        plot.axes = grid[i].axes
        plot.cax = grid.cbar_axes[i]

    p._setup_plots()
    fig.set_size_inches(figsize)
    
    for i, field in enumerate(fields):
        grid[i].axes.get_xaxis().get_major_formatter().set_scientific(False)
        grid[i].axes.get_yaxis().get_major_formatter().set_scientific(False)

    return p
