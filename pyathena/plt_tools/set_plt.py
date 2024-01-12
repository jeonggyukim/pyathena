import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

def toggle_xticks(axes, visible=False):
    axes = np.atleast_1d(axes)
    plt.setp([ax.get_xticklabels() for ax in axes], visible=visible)

def toggle_yticks(axes, visible=False):
    axes = np.atleast_1d(axes)
    plt.setp([ax.get_yticklabels() for ax in axes], visible=visible)

def set_plt_default():

    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.constrained_layout.use'] = True

    plt.rcParams['font.size'] = 15
    #plt.rcParams['font.weight'] = 300

    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['lines.linewidth'] = 2

    # plt.rcParams['xtick.top'] = True
    # plt.rcParams['ytick.right'] = True

def set_plt_fancy():
    # Chang-Goo's fancystyle
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.solid_capstyle'] = 'butt'

    plt.rcParams['legend.fancybox'] = True

    plt.rcParams['axes.prop_cycle'] = (cycler(color=['#008fd5',
                                                     '#fc4f30',
                                                     '#e5ae38',
                                                     '#6d904f',
                                                     '#fe01b1',
                                                     '#06c2ac',
                                                     '#fe019a', # neon pink
                                                     '#810f7c',
                                                     '#8b8b8b', # gray
                                                     'black']) +
                                       cycler(linestyle=['-','-','-','-','-',
                                                         '-','-','-','-','-']))

    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.titlesize'] = 'large'

    plt.rcParams['patch.edgecolor'] = 'f0f0f0'
    plt.rcParams['patch.linewidth'] = 0.5

    plt.rcParams['svg.fonttype'] = 'path'

    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 1.0
    plt.rcParams['grid.color'] = 'cbcbcb'

    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.minor.visible'] = False

    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True

    plt.rcParams['font.size'] = 15.0

    plt.rcParams['savefig.edgecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    # plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams['figure.subplot.left'] = 0.08
    # plt.rcParams['figure.subplot.right'] = 0.95
    # plt.rcParams['figure.subplot.bottom'] = 0.07
    # plt.rcParams['figure.subplot.hspace'] = 0.0
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.figsize'] = (8, 6)
    # plt.rcParams['figure.dpi'] = 200
