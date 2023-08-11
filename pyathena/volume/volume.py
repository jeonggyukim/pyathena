import numpy as np
import matplotlib.pyplot as plt
import yt
from yt.visualization.volume_rendering.api import Scene, create_volume_source

def get_mytf(b, c, nlayer=0):
    """set yt transfer functions using a colormat for
    descrete layers or linramp

    Parameters
    ==========
        b : tuple, list
            data bounds (min, max)
        c : str
            colormap name (yt understandable colormap name)
        nlayer : int (optional)
            number of layers, if 0, linramp function will be used to
            linearly map colormap

    Retruns
    =======
        tf : yt.visualization.volume_rendering.transfer_functions.ColorTransferFunction
    """
    def linramp(vals, minval, maxval):
        return 0.5 * (vals - vals.min()) / (vals.max() - vals.min()) + 0.5
    tf = yt.ColorTransferFunction((np.log10(b[0]), np.log10(b[1])))
    if nlayer == 0:
        tf.map_to_colormap(
            np.log10(b[0]), np.log10(b[1]), scale=20, scale_func=linramp, colormap=c
        )
    else:
        tf.add_layers(nlayer, w=0.1, alpha=np.linspace(10,40,nlayer), colormap=c)

    return tf

def render_volume(ds, f, b, c, nlayer=0, render = True):
    """Initial volume renderinf for a given field

    Parameters
    ==========
        ds : yt.DataSet
        f : str
            field name to render
        b : tuple, list
            data bounds (min, max)
        c : str
            colormap name (yt understandable colormap name)
        nlayer : int
            number of layers, if 0, linramp function will be used to
            linearly map colormap
        render : bool
            if True, the scene will be actually rendered (take time)
    Returns
    =======
        im : yt.data_objects.image_array.ImageArray
            rendered image if render=True, None otherwise
        tf : yt.visualization.volume_rendering.transfer_functions.ColorTransferFunction
        sc : yt.visualization.volume_rendering.scene.Scene
    """
    sc = yt.create_scene(ds, field=f)

    tf = get_mytf(b, c, nlayer=nlayer)
    sc[0].set_transfer_function(tf)

    cam = sc.camera
    cam.set_position([1024,-512,1024],north_vector=[0,0,1])
    cam.zoom(1.5)
    cam.set_resolution(1024)
    sc.annotate_domain(ds,color=[1,1,1,1])
    if render:
        im = sc.render()
    else:
        im = None

    return im, tf, sc

def add_volume_source(sc, ds, f, b, c, nlayer=0, render=True):
    """Add additional volume source to the Scene
    """
    vol = create_volume_source(ds, field=f)
    tf = get_mytf(b, c, nlayer=nlayer)
    vol.set_transfer_function(tf)

    sc.add_source(vol)
    if render:
        im = sc.render()
    else:
        im = None

    return im, tf

def add_tf_to_image(fig, ds, f, tf, xoff=0.1):
    """Add transfer function colorbar to image
    """
    ax2 = fig.add_axes([1-xoff,0.1,0.05,0.8])
    tf.vert_cbar(256,False,ax2,label_fmt="%d")
    if f[1].startswith('xray'):
        label = f"log ${ds.field_info[f].display_name.replace('$','')}\,[{ds.field_info[f].units}]$"
    else:
        label = f"log {ds.field_info[f].display_name} [{ds.field_info[f].units}]"
    ax2.set_ylabel(label,weight='bold',fontsize=15)

def save_with_tf(ds, f, im, tf, fout = None, xoff = 0.1, timestamp=True):
    """Save image with a transfer function colorbar
    """
    fig = plt.figure(figsize=(5,5),dpi=200)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    ax.imshow(im.swapaxes(0,1))
    add_tf_to_image(fig, ds, f, tf, xoff=xoff)
    if timestamp:
        ax.annotate(f"t={ds.current_time.to('Myr').v:5.1f} Myr",(xoff,0.95),
                    xycoords='axes fraction',ha='left',va='top',weight='bold')
    if fout is not None:
        fig.savefig(fout,dpi=200,bbox_inches='tight')
    print(f'file saved: {fout}')

    return fig
