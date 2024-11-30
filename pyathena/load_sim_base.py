from abc import ABC

class LoadSimBase(ABC):
    """Common properties to all LoadSim classes

    Parameters
    ----------

    Attributes
    ----------
    basedir : str
        Directory where simulation output files are stored.
    savdir : str
        Directory where pickles and figures are saved.
    basename : str
        basename (last component) of `basedir`
    load_method : str
        Load vtk/hdf5 snapshots using 'pyathena', 'pythena_classic' (vtk only), or
        'yt'. Defaults to 'pyathena'.
    """

    @property
    def basedir(self):
        return self._basedir

    @property
    def basename(self):
        return self._basename

    @property
    def savdir(self):
        return self._savdir

    @savdir.setter
    def savdir(self, value):
        if value is None:
            self._savdir = self._basedir
        else:
            self._savdir = value

    @property
    def load_method(self):
        return self._load_method

    @savdir.setter
    def load_method(self, value):
        if value in ['pyathena', 'pyatheha_classic', 'yt']:
            self._load_method = value
        else:
            raise ValueError('Unrecognized load_method: ', value)
