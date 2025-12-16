import numpy as np
import xarray as xr
import astropy.constants as ac
import astropy.units as au
from ..util.derivative import gradient

# utilities for CR rotation
def get_b_angle(Bcc1,Bcc2,Bcc3,tiny=1.e-7):
    B_R = np.sqrt(Bcc1**2+Bcc2**2)
    B_r = np.sqrt(B_R**2+Bcc3**2)
    sint_b=xr.where(B_r<tiny,xr.ones_like(B_r),B_R/B_r)
    cost_b=xr.where(B_r<tiny,xr.zeros_like(B_r),Bcc3/B_r)
    sinp_b=xr.where(B_R<tiny,xr.zeros_like(B_R),Bcc2/B_R)
    cosp_b=xr.where(B_R<tiny,xr.ones_like(B_R),Bcc1/B_R)

    return sint_b,cost_b,sinp_b,cosp_b

def rotate_vector(angles,vector):
    sint,cost,sinp,cosp=angles
    v1,v2,v3=vector
    # First apply R1, then apply R2
    newv1 =  cosp * v1 + sinp * v2
    v2 = -sinp * v1 + cosp * v2

    # now apply R2
    v1 =  sint * newv1 + cost * v3
    v3 = -cost * newv1 + sint * v3

    return v1,v2,v3

def inverse_rotate_vector(angles,vector):
    sint,cost,sinp,cosp=angles
    v1,v2,v3=vector
    # First apply R2^-1, then apply R1^-1
    newv1 = sint * v1 - cost * v3
    v3 = cost * v1 + sint * v3

    # now apply R1^-1
    v1 = cosp * newv1 - sinp * v2
    v2 = sinp * newv1 + cosp * v2

    return v1,v2,v3

class PostProcessingZprof:
    def Fcr_parallel(self, data):
        """CR flux along B field direction
        """
        # B vector for angle
        Bcc = [data[f] for f in ["Bcc1","Bcc2","Bcc3"]]

        # vector to rotate
        Fc = [data[f] for f in ["0-Fc1","0-Fc2","0-Fc3"]]

        # rotation
        angles = get_b_angle(*Bcc)
        Fc_rot = rotate_vector(angles,Fc)

        # final results
        return Fc_rot[0]

    def GradPcr_parallel_direct(self, data):
        """CR pressure gradient along B field direction (non-steady state))
        """
        # B vector for angle
        Bcc = [data[f] for f in ["Bcc1","Bcc2","Bcc3"]]

        # vector to rotate
        # Finite difference gradient
        gradPcr = gradient(data['0-Ec'].data/3.0,
                           data.x.data,data.y.data,data.z.data)

        # rotation
        angles = get_b_angle(*Bcc)
        gradPcr_rot = rotate_vector(angles,gradPcr)

        # final results
        return gradPcr_rot[0]

    def Fcr_diff_parallel(self, data):
        """CR diffusion flux along B field direction
        """
        # B vector for angle
        Bcc = [data[f] for f in ["Bcc1","Bcc2","Bcc3"]]

        # vector to rotate
        vdiff = [data[f] for f in ["0-Vd1","0-Vd2","0-Vd3"]]

        # other variables
        ec = data["0-Ec"]

        # rotation
        angles = get_b_angle(*Bcc)
        vdiff_rot = rotate_vector(angles,vdiff)

        # final results
        fd_b = 4./3.*ec*np.abs(vdiff_rot[0])
        return fd_b

    def Gamma_cr_stream(self, data):
        """CR streaming heating along B field direction
           (v_s,perp is zero if B is turned on)
        """
        # vmax in code units
        vlim = self.par["cr"]["vmax"]/self.u.velocity.cgs.value
        invlim = 1/vlim

        # B vector for angle
        Bcc = [data[f] for f in ["Bcc1","Bcc2","Bcc3"]]

        # vector to rotate
        fc = [data[f]*vlim for f in ["0-Fc1","0-Fc2","0-Fc3"]]
        vel = [data[f] for f in ["vel1","vel2","vel3"]]
        vs = [data[f] for f in ["0-Vs1","0-Vs2","0-Vs3"]]

        # other variables
        ec = data["0-Ec"]
        sigma_diff = data["0-Sigma_diff1"]
        sigma_adv = data["0-Sigma_adv1"]

        # rotation
        angles = get_b_angle(*Bcc)
        fc_rot = rotate_vector(angles,fc)
        vel_rot = rotate_vector(angles,vel)
        vs_rot = rotate_vector(angles,vs)

        # final results
        sigma_para = invlim/(1.0/sigma_diff + 1.0/sigma_adv)
        heating = -vs_rot[0]*sigma_para*(fc_rot[0]-4.0/3.0*ec*vel_rot[0])

        return heating

    def GradPcr_parallel(self,data):
        """CR pressure gradient along B field direction (assuming steady state flux)
        """
        # vmax in code units
        vlim = self.par["cr"]["vmax"]/self.u.velocity.cgs.value

        fd_b = self.Fcr_diff_parallel(data)
        sigma_diff = data["0-Sigma_diff1"]
        kappac = vlim/sigma_diff

        # final results
        return fd_b/kappac

    def CRLosses(self, data, ng=0):
        # this is done for single-bin approximation
        eV_erg = 1.6021773e-12

        # everything in code units
        ekin_bin = 1e9 # GeV
        ekin_bin *= eV_erg / self.u.erg # code units
        speed_of_light = ac.c.cgs/self.u.velocity.cgs
        hydrogen_mass = self.u.mH/self.u.mass.cgs
        p_bin = np.sqrt((ekin_bin/speed_of_light)**2 + 2*ekin_bin*hydrogen_mass)
        p = [p_bin]
        erel = [np.sqrt((p[0]*speed_of_light)**2 + hydrogen_mass**2*speed_of_light**4)]
        ekin = [erel[0] - hydrogen_mass*speed_of_light**2]
        vp = [speed_of_light * np.sqrt(1-(hydrogen_mass * speed_of_light**2/erel[0])**2)]

        # convert to cgs
        erel_erg = erel[ng]*self.u.erg
        erel_ev = erel_erg/eV_erg
        ekin_ev = ekin[ng]*self.u.erg / eV_erg
        vp_cgs = vp[ng]*self.u.cm/self.u.s

        # ionization losses
        lambdac = 1.21 * 1.27e-15*(ekin_ev/1e6)**(-0.82)*vp_cgs/erel_ev
        # hadronic losses
        if (ekin_ev > 1e10):
            lambdac += 1.18 * 3.85e-16 * (erel_ev/1e9)**(0.28)*(erel_ev/1e9 + 200)**(-0.2)
        elif (ekin_ev > 0.28e9):
            lambdac +=  1.18 * 2.82e-15 * (ekin_ev/1e10)**(1.28)/(erel_ev/1e9)

        # convert to code units
        lambdac *= self.u.s

        # final results
        ec = data["0-Ec"]
        rho = data["rho"]
        density_to_nH = (self.u.density.cgs/(self.u.mH*self.u.muH/au.cm**3))
        nH = rho * density_to_nH
        return -nH * ec * lambdac

    def CRwork(self, data, ng=0, split=False):
        # vmax in code units
        vlim = self.par["cr"]["vmax"]/self.u.velocity.cgs.value
        invlim = 1/vlim

        # B vector for angle
        Bcc = [data[f] for f in ["Bcc1","Bcc2","Bcc3"]]

        # vector to rotate
        fc = [data[f]*vlim for f in ["0-Fc1","0-Fc2","0-Fc3"]]
        vel = [data[f] for f in ["vel1","vel2","vel3"]]
        vs = [data[f] for f in ["0-Vs1","0-Vs2","0-Vs3"]]
        vd = [data[f] for f in ["0-Vd1","0-Vd2","0-Vd3"]]

        # other variables
        ec = data["0-Ec"]
        sigma_diff = data["0-Sigma_diff1"]
        sigma_adv = data["0-Sigma_adv1"]

        # rotation
        angles = get_b_angle(*Bcc)
        fc_rot = rotate_vector(angles,fc)
        vel_rot = rotate_vector(angles,vel)
        # vs_rot = rotate_vector(angles,vs)
        vd_rot = rotate_vector(angles,vd)

        # final results
        sigma_para = invlim/(1.0/sigma_diff + 1.0/sigma_adv)
        sigma_perp = sigma_diff*self.par["cr"]["perp_to_par_diff"]
        kappa_perp = vlim/sigma_perp
        work_para = -vel_rot[0]*sigma_para*(fc_rot[0]-4.0/3.0*ec*vel_rot[0])
        work_perp = -4.0/3.0*ec/kappa_perp*(vd_rot[1]*vel_rot[1] + vd_rot[2]*vel_rot[2])

        if split:
            return work_para, work_perp
        else:
            return work_para + work_perp

    def EffecitveCRvelocity(self, data, dir=1):
        """Effective CR velocity along B field direction
        """
        # vmax in code units
        vlim = self.par["cr"]["vmax"]/self.u.velocity.cgs.value

        ec = data["0-Ec"]
        return data[f"0-Fc{dir}"]*vlim/(4.0/3.0*ec)

    def GetAreaForPhaseAndVz(self, data, phase, np=0, vz_dir=1):
        """Area of the face where vz_dir*Vz>0 and phase=np
        np=0: cold, 1: cool, 2: warm, 3: ionized, 4: hot
        vz_dir=1: top face, vz_dir=-1: bottom face
        """
        # This function requires pmb pointer, so it should be implemented in C++
        # Here is the reference implementation in C++:
        #
        area = self.domain["dx"][0]*self.domain["dx"][1]
        return area*((vz_dir*data["vel3"] > 0) & (phase == np))

    def get_all_crprop(self, data, ng=0):
        results = xr.Dataset()
        results["Fcr_diff_parallel"] = self.Fcr_diff_parallel(data)
        results["Gamma_cr_stream"] = self.Gamma_cr_stream(data)
        results["GradPcr_parallel"] = self.GradPcr_parallel(data)
        results["CRLosses"] = self.CRLosses(data, ng=ng)
        results["CRwork_total"] = self.CRwork(data, ng=ng, split=False)
        results["CRwork_para"] = self.CRwork(data, ng=ng, split=True)[0]
        results["CRwork_perp"] = self.CRwork(data, ng=ng, split=True)[1]
        results["0-Veff1"] = self.EffecitveCRvelocity(data, dir=1)
        results["0-Veff2"] = self.EffecitveCRvelocity(data, dir=2)
        results["0-Veff3"] = self.EffecitveCRvelocity(data, dir=3)

        return results

    def construct_zprof(self, data):
        phase = self.set_phase(data)
        phname = ["CNM","UNM","WNM","WHIM","HIM"]
        zplist = []
        zprof_data = xr.Dataset()
        # density field for sanity check
        if "rho" in data:
            zprof_data["rho"] = data["rho"]
        # add CR properties
        if self.options["cosmic_ray"]:
            zprof_data.update(self.get_all_crprop(data, ng=0))
        if "cool_rate" not in data:
            self.add_coolheat(data)
        zprof_data["cool_rate"] = data["cool_rate"]
        zprof_data["heat_rate"] = data["heat_rate"]
        for vz_dir in [-1,1]:
            zplist_ = []
            for np in range(5):
                area = self.GetAreaForPhaseAndVz(data, phase, np=np, vz_dir=vz_dir)
                zprof = (zprof_data*area).sum(dim=["x","y"]).assign_coords(phase=phname[np])
                zplist_.append(zprof)
            zplist.append(xr.concat(zplist_, dim="phase").assign_coords(vz_dir=vz_dir))

        zprof = xr.concat(zplist, dim="vz_dir")
        return zprof