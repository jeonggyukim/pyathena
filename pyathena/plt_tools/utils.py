def texteffect(fontsize=12, linewidth=3, foreground='w'):
  try:
    from matplotlib.patheffects import withStroke
    myeffect = withStroke(foreground=foreground, linewidth=linewidth)
    kwargs = dict(path_effects=[myeffect], fontsize=fontsize)
  except ImportError:
    kwargs = dict(fontsize=fontsize)

  return kwargs
