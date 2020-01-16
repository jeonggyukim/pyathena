import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_my_cmap(name='Blues'):
    NN = np.linspace(0, 1, 256)
    cm_orig = mpl.cm.get_cmap(name)
    cm = cm_orig(NN)
    for i in range(cm.shape[0]//4):
        cm[i, 3] = NN[i*4]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_' + name, cm, 256)

    return cmap
