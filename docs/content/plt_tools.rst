==============
Plotting Tools
==============

The ``pyathena.plt_tools`` module provides matplotlib utilities for
visualizing simulation data.

Colormaps
=========

.. autofunction:: pyathena.plt_tools.cmap_shift.cmap_shift
.. autofunction:: pyathena.plt_tools.cmap.cmap_apply_alpha

Annotations
===========

.. autoclass:: pyathena.plt_tools.line_annotation.LineAnnotation
    :members:

Plots
=====

.. autofunction:: pyathena.plt_tools.plt_joint_pdf.plt_joint_pdf
.. autofunction:: pyathena.plt_tools.plt_starpar.scatter_sp
.. autofunction:: pyathena.plt_tools.plt_starpar.legend_sp
.. autofunction:: pyathena.plt_tools.plt_starpar.colorbar_sp

Style
=====

.. autofunction:: pyathena.plt_tools.set_plt.set_plt_default
.. autofunction:: pyathena.plt_tools.set_plt.set_plt_fancy
.. autofunction:: pyathena.plt_tools.set_plt.toggle_xticks
.. autofunction:: pyathena.plt_tools.set_plt.toggle_yticks
.. autofunction:: pyathena.plt_tools.utils.texteffect
.. autofunction:: pyathena.plt_tools.make_spines_invisible.make_patch_spines_invisible

Movies
======

.. autofunction:: pyathena.plt_tools.make_movie.make_movie
.. autofunction:: pyathena.plt_tools.make_movie.display_movie
