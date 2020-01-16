
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['figure.dpi'] = 150

plt.rcParams['font.size'] = 15
plt.rcParams['font.weight'] = 300

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

def toggle_xticks(axes, visible=False):
    axes = np.atleast_1d(axes)
    plt.setp([ax.get_xticklabels() for ax in axes], visible=visible)

def toggle_yticks(axes, visible=False):
    axes = np.atleast_1d(axes)
    plt.setp([ax.get_yticklabels() for ax in axes], visible=visible)
