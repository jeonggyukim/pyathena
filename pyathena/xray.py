import yt
import numpy as np
import pyxsim
import soxs
import os, glob
import xarray as xr
import matplotlib.pyplot as plt


class Xray(object):
    def __init__(self, sim, ytds, verbose=True):
        self.sim = sim
        self.ytds = ytds
        self.basename = f"{ytds}"
        self.savdir = os.path.join(sim.savdir, "xray")
        self.verbose = verbose
        os.makedirs(self.savdir, exist_ok=True)

        self.par = sim.par

    def add_ytfields(self):
        from yt.utilities.physical_constants import mh, kboltz

        muH = self.sim.u.muH

        def _ndensity(field, data):
            return data[("gas", "density")] / (muH * mh)

        def _nelectron(field, data):
            return data[("gas", "density")] * data[("athena", "xe")] / (muH * mh)

        if ("athena", "temperature") in self.ytds.field_list:

            def _temperature(field, data):
                return data[("athena", "temperature")] * yt.units.K

        else:

            def _temperature(field, data):
                mu = muH / (1.1 + data[("athena", "xe")] - data[("athena", "xH2")])
                return (
                    data[("gas", "pressure")]
                    / (data[("gas", "density")])
                    * (mu * mh / kboltz)
                )

        def _EM(field, data):
            return (
                data[("gas", "H_nuclei_density")]
                * data[("gas", "El_number_density")]
                * data[("gas", "cell_volume")]
            )

        # add/override fields
        ds = self.ytds
        ds.add_field(
            ("gas", "H_nuclei_density"),
            function=_ndensity,
            force_override=True,
            units="cm**(-3)",
            display_name=r"$n_{\rm H}$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "El_number_density"),
            function=_nelectron,
            force_override=True,
            units="cm**(-3)",
            display_name=r"$n_{\rm e}$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "temperature"),
            function=_temperature,
            force_override=True,
            units="K",
            display_name=r"$T$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "emission_measure"),
            function=_EM,
            force_override=True,
            units="cm**(-3)",
            display_name=r"EM",
            sampling_type="cell",
        )

    def add_xray_fields(self, xf_emin=0.5, xf_emax=7.0, **kwargs):
        """Add xray fields by using pyxsim.CIESourceModel

        Parameters
        ==========
        xf_emin : float, array
            minimum energy in keV of X-ray field to be added
        xf_emax : float, array
            maximum energy in keV of X-ray field to be added
        model : string
            Which spectral emission model to use. Accepts either "apec", "spex",
            "mekal", or "cloudy".
        emin : float
            The minimum energy for the spectrum in keV.
        emax : float
            The maximum energy for the spectrum in keV.
        nbins : integer
            The number of channels in the spectrum.
        Zmet : float, string, or tuple of strings
            The metallicity. If a float, assumes a constant metallicity throughout
            in solar units. If a string or tuple of strings, is taken to be the
            name of the metallicity field.

        Additional parameters for CIESourceModel is needed (see https://hea-www.cfa.harvard.edu/~jzuhone/pyxsim/api/source_models.html#pyxsim.source_models.thermal_sources.CIESourceModel)

        """
        if "Z_gas" in self.par["problem"]:
            Zmet = self.par["problem"]["Z_gas"]
        else:
            Zmet = 1.0

        self.source_model = pyxsim.CIESourceModel(**kwargs)
        xray_fields = []
        for emin, emax in zip(np.atleast_1d(xf_emin), np.atleast_1d(xf_emax)):
            xray_fields += self.source_model.make_source_fields(self.ytds, emin, emax)
        self.xray_fields = xray_fields

    def create_profile(self):
        if not hasattr(self, "xray_fields"):
            self.add_xray_fields()
        fullbox = self.ytds.all_data()
        Nx, Ny, Nz = self.ytds.domain_dimensions
        le = self.ytds.domain_left_edge.v
        re = self.ytds.domain_right_edge.v

        profile = yt.create_profile(
            data_source=fullbox,
            bin_fields=[("gas", "temperature"), ("gas", "velocity_z")],
            fields=[("gas", "volume"), ("gas", "mass")] + self.xray_fields,
            n_bins=(256, 384),
            units=dict(temperature="K", velocity_z="km/s", volume="pc**3", mass="Msun"),
            logs=dict(radius=False, velocity_z=False),
            weight_field=None,
            extrema=dict(temperature=(10, 1.0e10), velocity_z=(-1536, 1536)),
        )

        return profile

    def convert_profile_to_dataset(self, profile):
        # convert profiles to xarray dataset
        dset = xr.Dataset()
        for (g, k), v in profile.items():
            x = 0.5 * (profile.x_bins[:-1] + profile.x_bins[1:])
            y = 0.5 * (profile.y_bins[:-1] + profile.y_bins[1:])
            g, xf = profile.x_field
            g, yf = profile.y_field
            da = xr.DataArray(v, coords=[x, y], dims=[xf, yf])
            dset[k] = da
        return dset

    def get_profile(self, overwrite=False):
        fname = os.path.join(self.savdir, f"{self.ytds}.profile.nc")
        if os.path.isfile(fname):
            if overwrite:
                os.remove(fname)
            else:
                dset = xr.open_dataset(fname)
                return dset

        profile = self.create_profile()
        dset = self.convert_profile_to_dataset(profile)
        dset.to_netcdf(fname)

        return dset

    def make_photons(
        self, exp_time=(100, "ks"), area=(1, "m**2"),
        redshift=1.19e-5, overwrite=False
    ):
        ds = self.ytds

        from yt.utilities.cosmology import Cosmology
        cosmo = Cosmology()
        dist_kpc = cosmo.angular_diameter_distance(0.0, redshift).to("kpc")

        photon_fname = os.path.join(self.savdir, f"{ds}_{int(dist_kpc):d}kpc_photons.h5")

        box = ds.all_data()  # for the full box
        if not overwrite and os.path.isfile(photon_fname):
            if self.verbose:
                print("photon file {} exist".format(photon_fname))
        else:
            _photons, n_cells = pyxsim.make_photons(
                photon_fname,
                box,
                redshift,
                area,
                exp_time,
                self.source_model
            )
        self.photon_fname = photon_fname

    def project_photons(
        self, axis="z", NH=0.02, sky_center_src=(45.0, 30.0), overwrite=False
    ):
        ds = self.ytds
        self.sky_center = sky_center_src

        if not hasattr(self, "photon_fname"):
            print("create photon list first: call make_photons")
            return

        event_fname = self.photon_fname.replace("photons", "{}_events".format(axis))
        if not overwrite and os.path.isfile(event_fname):
            if self.verbose:
                print("event file {} exist".format(event_fname))
        else:
            n_events = pyxsim.project_photons(
                self.photon_fname,
                event_fname,
                axis,
                sky_center_src,
                absorb_model="tbabs",
                nH=NH,
            )
        self.event_fname = event_fname

    def create_simput(self):
        ds = self.ytds
        if not hasattr(self, "event_fname"):
            print("create event file first: call project_photons")
            return
        events = pyxsim.EventList(self.event_fname)
        events.write_to_simput(
            self.event_fname.replace("_events.h5", ""), overwrite=True
        )
        self.simput_fname = self.event_fname.replace("_events.h5", "_simput.fits")

    def instrument_simulator(self, exp=100, inst="lem_2eV", overwrite=False):
        if not hasattr(self, "simput_fname"):
            self.create_simput()
        simput_fname = self.simput_fname
        for target in ["src", "onsrc", "offsrc"]:
            if target == "src":
                sky_center = self.sky_center
                kwargs = dict(instr_bkgnd=False, foreground=False, ptsrc_bkgnd=False)
            elif target == "onsrc":
                sky_center = self.sky_center
                kwargs = dict(instr_bkgnd=True, foreground=True, ptsrc_bkgnd=True)
            elif target == "offsrc":
                sky_center = (0, 0)
                kwargs = dict(instr_bkgnd=True, foreground=True, ptsrc_bkgnd=True)
            evt_fname = self.simput_fname.replace("simput", f"{target}_{exp}ks_{inst}")
            if not overwrite and os.path.isfile(evt_fname):
                if self.verbose:
                    print("source event file {} exist".format(evt_fname))
            else:
                soxs.instrument_simulator(
                    simput_fname,
                    evt_fname,
                    (exp, "ks"),
                    inst,
                    sky_center,
                    overwrite=True,
                    **kwargs,
                )
            if os.path.isfile(evt_fname):
                soxs.write_image(
                    evt_fname,
                    evt_fname.replace(".fits", "_img.fits"),
                    emin=0.1,
                    emax=2.0,
                    overwrite=True,
                )
                soxs.write_spectrum(
                    evt_fname, evt_fname.replace(".fits", "_spec.pha"), overwrite=True
                )

    def find_img_files(self):
        return glob.glob(os.path.join(self.savdir, "{}*_img.fits".format(self.ytds)))

    def find_spec_files(self):
        return glob.glob(os.path.join(self.savdir, "{}*_spec.pha".format(self.ytds)))

    def parse_filename(self, f):
        sp = os.path.basename(f.replace(self.basename, "")).split("_")
        _, dist, axis, src, exptime = sp[:5]
        inst = "_".join(sp[5:-1])
        return dist, axis, src, exptime, inst

    def select_files(self, files, match=dict(exptime="100ks")):
        flist = []
        for f in files:
            par = self.parse_filename(f)
            skip = False
            for k, v in match.items():
                if v != par[k]:
                    skip = True
            if skip:
                continue
            flist.append(f)

        return sorted(flist)

    def plot_image(
        self,
        img_file,
        hdu="IMAGE",
        stretch="linear",
        vmin=None,
        vmax=None,
        facecolor="black",
        center=None,
        width=None,
        fig=None,
        cmap=None,
        cbar=True,
        grid_spec=None,
    ):
        """
        Plot a FITS image created by SOXS using Matplotlib.

        Parameters
        ----------
        img_file : str
            The on-disk FITS image to plot.
        hdu : str or int, optional
            The image extension to plot. Default is "IMAGE"
        stretch : str, optional
            The stretch to apply to the colorbar scale. Options are "linear",
            "log", and "sqrt". Default: "linear"
        vmin : float, optional
            The minimum value of the colorbar. If not set, it will be the minimum
            value in the image.
        vmax : float, optional
            The maximum value of the colorbar. If not set, it will be the maximum
            value in the image.
        facecolor : str, optional
            The color of zero-valued pixels. Default: "black"
        center : array-like
            A 2-element object giving an (RA, Dec) coordinate for the center
            in degrees. If not set, the reference pixel of the image (usually
            the center) is used.
        width : float, optional
            The width of the image in degrees. If not set, the width of the
            entire image will be used.
        figsize : tuple, optional
            A 2-tuple giving the size of the image in inches, e.g. (12, 15).
            Default: (10,10)
        cmap : str, optional
            The colormap to be used. If not set, the default Matplotlib
            colormap will be used.

        Returns
        -------
        A tuple of the :class:`~matplotlib.figure.Figure` and the
        :class:`~matplotlib.axes.Axes` objects.
        """
        from astropy.io import fits
        from astropy.visualization.wcsaxes import WCSAxes
        from astropy import wcs
        from astropy.wcs.utils import proj_plane_pixel_scales
        from matplotlib.colors import LogNorm, Normalize, PowerNorm

        if stretch == "linear":
            norm = Normalize(vmin=vmin, vmax=vmax)
        elif stretch == "log":
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif stretch == "sqrt":
            norm = PowerNorm(0.5, vmin=vmin, vmax=vmax)
        else:
            raise RuntimeError(f"'{stretch}' is not a valid stretch!")
        with fits.open(img_file) as f:
            hdu = f[hdu]
            w = wcs.WCS(hdu.header)
            pix_scale = proj_plane_pixel_scales(w)
            if center is None:
                center = w.wcs.crpix
            else:
                center = w.wcs_world2pix(center[0], center[1], 0)
            if width is None:
                dx_pix = 0.5 * hdu.shape[0]
                dy_pix = 0.5 * hdu.shape[1]
            else:
                dx_pix = width / pix_scale[0]
                dy_pix = width / pix_scale[1]
            if fig is None:
                fig = plt.figure(figsize=(10, 10))
            if grid_spec is None:
                grid_spec = [0.15, 0.1, 0.8, 0.8]
            # fig = plt.figure(figsize=figsize)
            # ax = WCSAxes(fig, [0.15, 0.1, 0.8, 0.8], wcs=w)
            ax = fig.add_subplot(grid_spec, projection=w)
            im = ax.imshow(hdu.data, norm=norm, cmap=cmap)
            ax.set_xlim(center[0] - 0.5 * dx_pix, center[0] + 0.5 * dx_pix)
            ax.set_ylim(center[1] - 0.5 * dy_pix, center[1] + 0.5 * dy_pix)
            ax.set_facecolor(facecolor)
            if cbar:
                plt.colorbar(im)
        return fig, ax

    def show(self, source="src", width=0.2, vmin=0, vmax=100):
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(12, 10))

        gs = GridSpec(3, 3, figure=fig, height_ratios=[0.9, 1, 1])

        i = 0
        for f in sorted(self.find_img_files()):
            region, dist, axis, src, exptime, inst = self.parse_filename(f)
            if source != src:
                continue
            self.plot_image(
                f,
                stretch="sqrt",
                cmap="arbre",
                width=width,
                vmin=vmin,
                vmax=vmax,
                fig=fig,
                grid_spec=gs[0, i],
                cbar=i == 2,
            )
            plt.title(region)
            i += 1

        ax = fig.add_subplot(gs[1, :])
        for f in sorted(self.find_spec_files()):
            region, dist, axis, src, exptime, inst = self.parse_filename(f)
            if src != source:
                continue
            fig, ax = soxs.plot_spectrum(
                f, xmin=0.1, xmax=2.0, fig=fig, ax=ax, fontsize=None, label=region
            )
        plt.legend()

        for j, xlim in enumerate([(0.55, 0.6), (0.8, 0.85), (0.975, 1.025)]):
            ax = fig.add_subplot(gs[2, j])
            for f in sorted(self.find_spec_files()):
                region, dist, axis, src, exptime, inst = self.parse_filename(f)
                if src != source:
                    continue
                fig, ax = soxs.plot_spectrum(
                    f, xmin=0.1, xmax=2.0, fontsize=None, fig=fig, ax=ax, label=region
                )
            plt.xlim(xlim)

        return fig

    def do_all(self):
        with plt.style.context(
            {"figure.dpi": 150, "font.size": 10, "figure.figsize": (4, 3)}
        ):
            fig = plt.figure()
            self.show_profile()
            fig.savefig(
                os.path.join(self.profdir, "{}_xray_profile.png".format(self.basename)),
                bbox_inches="tight",
            )
            plt.close(fig)
        self.project_photons(axis="z")
        self.instrument_simulator()
        with plt.style.context({"figure.dpi": 150, "font.size": 10}):
            fig = self.show()
            fig.savefig(
                os.path.join(self.figdir, "{}_xray_figure.png".format(self.basename)),
                bbox_inches="tight",
            )
            plt.close(fig)
