import numpy as np
import astropy.units as au
import astropy.constants as ac
import labellines
import matplotlib.pyplot as plt


# calculate SNR properties as written in the TIGRESS++ document Section 7.4
class SNR(object):
    def __init__(self, mu=1.27, rfb_res=3, nmin=-5, nmax=5, Minj=20, Einj=1):
        self.nH = np.logspace(nmin, nmax, 100) / au.cm**3
        self.Minj = Minj * au.M_sun
        self.Einj = Einj * 1.0e51 * au.erg

        self.dx = 2 ** np.arange(6) * au.pc
        self.rfb = self.dx * rfb_res

        self.mu = 1.27
        self.gamma = 5 / 3.0

    def get_volume(self):
        Vsnr = 4 * np.pi / 3.0 * self.rfb**3
        return Vsnr[np.newaxis, :]

    def get_mass(self):
        rho = self.mu * ac.m_p * self.nH[:, np.newaxis]
        Vsnr = self.get_volume()
        Menc = (rho * Vsnr).to("Msun")
        Msnr = Menc + self.Minj
        return Msnr.to("Msun")

    def get_density(self):
        Msnr = self.get_mass()
        Vsnr = self.get_volume()
        return (Msnr / Vsnr).cgs

    def get_number_density(self):
        return (self.get_density() / self.mu / ac.m_p).cgs

    def get_pterminal(self):
        n0 = (self.get_number_density() * au.cm**3).cgs
        pterm = 2.8e5 * n0 ** (-0.17) * au.M_sun * au.km / au.s
        return pterm

    def get_Msf(self):
        n0 = (self.get_number_density() * au.cm**3).cgs
        Msf = 1.54e3 * n0 ** (-0.33) * au.M_sun
        return Msf

    def get_pressure(self):
        Vsnr = self.get_volume()
        gm1 = self.gamma - 1
        return 0.72 * self.Einj / Vsnr * gm1

    def get_temperature(self):
        Psnr = self.get_pressure()
        nsnr = self.get_number_density()
        return (Psnr / ac.k_B / nsnr).cgs

    def get_velocity_E(self):
        Msnr = self.get_mass()
        return np.sqrt(0.28 * 2.0 * self.Einj / Msnr).to("km/s")

    def get_velocity_p(self):
        psnr = self.get_pterminal()
        Msnr = self.get_mass()
        return (psnr / Msnr).to("km/s")

    def get_velocity(self):
        v_e = self.get_velocity_E()
        v_p = self.get_velocity_p()
        return 1 / (1 / v_e + 1 / v_p)

    def plot_mass(self, save=True):
        # Msnr
        nbkgd = self.nH
        Msnr = self.get_mass()
        plt.plot(nbkgd, Msnr)
        lines = plt.gca().lines
        for line, dx_ in zip(lines, self.dx.value):
            line.set_label(f"$\Delta x = {int(dx_)} {{\\rm pc}}$")
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel(r"$n_H\,[{\rm cm^{-3}}]$")
        plt.ylabel(r"$M_{\rm snr}\,[{\rm M_\odot}]$")
        labellines.labelLines(
            lines[: len(self.dx)], xvals=(1.0e3, 1.0e4), drop_label=True
        )

        # Msf
        Msf = self.get_Msf()
        plt.gca().set_prop_cycle(None)
        plt.plot(nbkgd, Msf, ls=":")
        line = plt.gca().lines[-1]
        line.set_label(r"$M_{\rm sf}$")
        labellines.labelLine(line, x=1.0e-3)

        if save:
            plt.savefig("Msnr.png", bbox_inches="tight")

    def plot_temperature(self, save=True):
        nbkgd = self.nH
        Tsnr = self.get_temperature()
        plt.plot(nbkgd, Tsnr)
        lines = plt.gca().lines
        for line, dx_ in zip(lines, self.dx.value):
            line.set_label(f"$\Delta x = {int(dx_)} {{\\rm pc}}$")
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel(r"$n_H\,[{\rm cm^{-3}}]$")
        plt.ylabel(r"$T_{\rm snr}\,[{\rm K}]$")
        labellines.labelLines(
            lines[: len(self.dx)], xvals=(1.0e3, 1.0e4), drop_label=True
        )

        if save:
            plt.savefig("Tsnr.png", bbox_inches="tight")

    def plot_velocity(self, save=True):
        nbkgd = self.nH
        vsnr = self.get_velocity()
        vsnr_E = self.get_velocity_E()
        vsnr_p = self.get_velocity_p()
        plt.plot(nbkgd, vsnr)
        lines = plt.gca().lines
        for line, dx_ in zip(lines, self.dx.value):
            line.set_label(f"$\Delta x = {int(dx_)} {{\\rm pc}}$")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(bottom=0.1)
        labellines.labelLines(
            lines[: len(self.dx)], xvals=(1.0e3, 1.0e4), drop_label=True
        )

        plt.gca().set_prop_cycle(None)
        plt.plot(nbkgd, vsnr_E, ls=":")
        plt.gca().lines[-1].set_label("energy")

        plt.gca().set_prop_cycle(None)
        plt.plot(nbkgd, vsnr_p, ls="--")
        plt.gca().lines[-1].set_label("momentum")

        plt.legend()

        plt.xlabel(r"$n_H\,[{\rm cm^{-3}}]$")
        plt.ylabel(r"$v_{\rm snr}\,[{\rm km/s}]$")

        if save:
            plt.savefig("vsnr.png", bbox_inches="tight")


if __name__ == "__main__":
    snr = SNR()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    plt.sca(axes[0])
    snr.plot_mass(False)
    plt.sca(axes[1])
    snr.plot_temperature(False)
    plt.sca(axes[2])
    snr.plot_velocity(False)
    plt.tight_layout()
    fig.savefig("snr.png", bbox_inches="tight", dpi=200)
