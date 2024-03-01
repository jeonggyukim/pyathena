import yt
import pyxsim
import soxs
import os
import glob
import xarray as xr
import matplotlib.pyplot as plt


class Xray(object):
    def __init__(
        self,
        sim,
        ytds,
        zcut=128,
        verbose=True,
        emin=0.1,
        emax=10,
        nbins=1000,
        redshift=0.00005,
        model="spex",
        binscale="log",
    ):
        self.sim = sim
        self.ytds = ytds
        self.basename = ytds.basename.replace(".tar", "")
        self.savdir = os.path.join(sim.savdir, "xray")
        self.figdir = os.path.join(sim.savdir, "xray", "figure")
        self.profdir = os.path.join(sim.savdir, "xray", "profile")
        self.zcut = zcut
        self.verbose = verbose
        os.makedirs(self.savdir, exist_ok=True)
        os.makedirs(self.figdir, exist_ok=True)
        os.makedirs(self.profdir, exist_ok=True)

        Zmet = self.sim.par["problem"]["Z_gas"]

        self.source_model = pyxsim.CIESourceModel(
            model, emin, emax, nbins, Zmet, binscale=binscale, abund_table="aspl"
        )

    def add_xray_fields(self):
        xray_fields = self.source_model.make_source_fields(self.ytds, 0.5, 2.0)
        xray_fields += self.source_model.make_source_fields(self.ytds, 0.5, 7.0)
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
            bin_fields=[("gas", "z"), ("gas", "velocity_z")],
            fields=[
                ("gas", "volume"),
                ("gas", "mass"),
                ("gas", "xray_emissivity_0.5_2.0_keV"),
                ("gas", "xray_emissivity_0.5_7.0_keV"),
            ],
            n_bins=(Nz, 256),
            units=dict(z="pc", velocity_z="km/s", volume="pc**3", mass="Msun"),
            logs=dict(radius=False, velocity_z=False),
            weight_field=None,
            extrema=dict(z=(le[2], re[2]), velocity_z=(-1536, 1536)),
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
        fname = os.path.join(
            self.savdir, self.ytds.basename.replace(".tar", ".profile.nc")
        )
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

    def set_regions(self):
        zcut = self.zcut
        ds = self.ytds
        le = ds.domain_left_edge.v
        re = ds.domain_right_edge.v
        fullbox = ds.box(le, re)
        topbox = ds.box([le[0], le[1], zcut], re)
        botbox = ds.box(le, [re[0], re[1], -zcut])

        self.regions = dict(full=fullbox, top=topbox, bot=botbox)

    def make_photons(
        self, exp_time=(100, "ks"), area=(1, "m**2"), redshift=0.00005, overwrite=False
    ):
        from astropy.cosmology import WMAP7

        dist_kpc = int(WMAP7.comoving_distance(redshift).to("kpc").value)

        ds = self.ytds
        if not hasattr(self, "regions"):
            self.set_regions()
        photon_fnames = dict()
        for name, box in self.regions.items():
            photon_fname = os.path.join(
                self.savdir, "{}_{}_{}kpc_photons.h5".format(ds, name, dist_kpc)
            )
            if not overwrite and os.path.isfile(photon_fname):
                if self.verbose:
                    print("photon file {} exist".format(photon_fname))
            else:
                _photons, n_cells = pyxsim.make_photons(
                    photon_fname, box, redshift, area, exp_time, self.source_model
                )
            photon_fnames[name] = photon_fname
        self.photon_fnames = photon_fnames

    def project_photons(self, axis="z", absorb_model="tbabs", nH=0.02, overwrite=False):
        # ds = self.ytds
        if not hasattr(self, "photon_fnames"):
            self.make_photons()
        event_fnames = dict()
        for boxname, photon_fname in self.photon_fnames.items():
            event_fname = photon_fname.replace("photons", "{}_events".format(axis))
            if not overwrite and os.path.isfile(event_fname):
                if self.verbose:
                    print("event file {} exist".format(event_fname))
            else:
                _ = pyxsim.project_photons(
                    photon_fname,
                    event_fname,
                    axis,
                    (45.0, 30.0),
                    absorb_model="tbabs",
                    nH=nH,
                )
            event_fnames[boxname] = event_fname
        self.event_fnames = event_fnames

    def create_simput(self):
        # ds = self.ytds
        if not hasattr(self, "event_fnames"):
            self.project_photons()
        simput_fnames = dict()
        for boxname, event_fname in self.event_fnames.items():
            events = pyxsim.EventList(event_fname)
            events.write_to_simput(
                event_fname.replace("_events.h5", ""), overwrite=True
            )
            simput_fnames[boxname] = event_fname.replace("_events.h5", "_simput.fits")
        self.simput_fnames = simput_fnames

    def instrument_simulator(self, exp=100, inst="lem_2eV", overwrite=False):
        if not hasattr(self, "simput_fnames"):
            self.create_simput()
        for boxname, simput_fname in self.simput_fnames.items():
            evt_fname = simput_fname.replace("simput", "src_{}ks_{}".format(exp, inst))
            if not overwrite and os.path.isfile(evt_fname):
                if self.verbose:
                    print("source event file {} exist".format(evt_fname))
            else:
                soxs.instrument_simulator(
                    simput_fname,
                    evt_fname,
                    (exp, "ks"),
                    inst,
                    [45.0, 30.0],
                    overwrite=True,
                    instr_bkgnd=False,
                    foreground=False,
                    ptsrc_bkgnd=False,
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

            evt_fname = simput_fname.replace(
                "simput", "srcbkg_{}ks_{}".format(exp, inst)
            )
            if not overwrite and os.path.isfile(evt_fname):
                if self.verbose:
                    print("source+bkgd event file {} exist".format(evt_fname))
            else:
                soxs.instrument_simulator(
                    simput_fname,
                    evt_fname,
                    (exp, "ks"),
                    inst,
                    [45.0, 30.0],
                    overwrite=True,
                    instr_bkgnd=True,
                    foreground=True,
                    ptsrc_bkgnd=True,
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
        _, region, dist, axis, src, exptime = sp[:6]
        inst = "_".join(sp[6:-1])
        return region, dist, axis, src, exptime, inst

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

        # from astropy.visualization.wcsaxes import WCSAxes
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

    def show_profile(self):
        import numpy as np

        zcut = self.zcut
        prof = self.get_profile()

        prof["xray_emissivity_0.5_2.0_keV"].sel(z=slice(-np.inf, -zcut)).sum(
            dim="z"
        ).plot(label="bot")
        prof["xray_emissivity_0.5_2.0_keV"].sum(dim="z").plot(label="full")
        prof["xray_emissivity_0.5_2.0_keV"].sel(z=slice(zcut, np.inf)).sum(
            dim="z"
        ).plot(label="top")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-30, 1.0e-20)

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
