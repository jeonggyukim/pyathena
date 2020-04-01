# virial.py

import numpy as np
import yt.units as yu
import pandas as pd

from ..load_sim import LoadSim

class Virial:

    def get_num_max_virial(self):
        h = self.read_hst()
        idx = h.Mstar> 0.0
        
        return int(1.5*h['time_code'][idx].iloc[0] / self.par['output1']['dt'])

    @LoadSim.Decorators.check_pickle
    def read_virial_all(self, prefix='virial_all', savdir=None, force_override=False):

        rr = dict()
        for i in range(self.get_num_max_virial()):
            print(i, end=' ')
            r = self.read_virial(num=i)
            if i == 0:
                for k in r.keys():
                    rr[k] = []

            for k in r.keys():
                rr[k].append(r[k])

        return pd.DataFrame(rr)
            
    
    @LoadSim.Decorators.check_pickle
    def read_virial(self, num, R_over_Rcl=1.8,
                    prefix='virial', savdir=None, force_override=False):
        
        ds = self.load_vtk(num, load_method='yt')
        ds = self.add_fields_virial(ds)

        Rcl = self.par['problem']['R_cloud']
        xc = np.array([0.0,0.0,0.0])
        sph = ds.sphere(xc, (1.1*R_over_Rcl*Rcl,"pc"))
        sph2 = ds.sphere(xc, (R_over_Rcl*Rcl,"pc"))
        surf = ds.surface(sph, "dist", (R_over_Rcl*Rcl, "pc"))

        # These are the coordinates of all triangle vertices in the surface
        triangles = surf.triangles
        # construct triange normal vectors
        w1 = surf.triangles[:,1] - surf.triangles[:,0]
        w2 = surf.triangles[:,1] - surf.triangles[:,2]
        # calculate triangle areas
        vector_area = np.cross(w1, w2)/2*(yu.pc**2).to('cm**2')

        rdotTM = np.column_stack([surf['rdotTM_{0:s}'.format(x)] for x in 'xyz'])*(yu.erg/yu.cm**2)
        rdotPi = np.column_stack([surf['rdotPi_{0:s}'.format(x)] for x in 'xyz'])*(yu.erg/yu.cm**2)
        rhovrsq = np.column_stack([surf['rhovrsq_{0:s}'.format(x)] for x in 'xyz'])*(yu.g/yu.s)

        res = dict()
        res['R_over_Rcl'] = R_over_Rcl
        res['time'] = ds.current_time.value.item()
        res['I_E'] = ((sph2['cell_volume']*sph2['rho_rsq']).sum()).to('g*cm**2')

        # Flux of momentum of inertial
        res['S_surf'] = 0.5*np.sum(vector_area*rhovrsq)

        # Surface integral of Reynolds stress
        res['T_surf'] = 0.5*np.sum(vector_area*rdotPi)

        # Surface integral of Maxwell stress
        res['M_surf'] = np.sum(vector_area*rdotTM)/(4.0*np.pi)
        # Alternative way
        # M_surf_ = surf.calculate_flux('rdotTM_x', 'rdotTM_y', 'rdotTM_z', fluxing_field="ones")
        # Magnetic energy (volume)
        res['M_vol'] = (sph2['cell_volume']*sph2['magnetic_energy']).sum().to('erg')
        res['M'] = res['M_vol'] + res['M_surf']

        # Thermal + kinetic energy
        res['T_thm'] = 1.5*((sph2['cell_volume']*sph2['pressure']).sum()).to('erg')
        res['T_turb'] = 0.5*((sph2['cell_volume']*sph2['density']*
                              sph2['velocity_magnitude']**2).sum()).to('erg')
        res['T'] = res['T_thm'] + res['T_turb']

        # Gravitational term (gravitational energy in the absence of external potential)
        res['W'] = (sph2['density']*
                    (sph2['x']*sph2['gravitational_potential_gradient_x'] +\
                     sph2['y']*sph2['gravitational_potential_gradient_y'] +\
                     sph2['z']*sph2['gravitational_potential_gradient_z'])*
                    sph2['cell_volume']).sum().to('erg')

        return res
        
        
    @staticmethod
    def add_fields_virial(ds):
        def _dist(field, data):
            c = data.get_field_parameter("center")
            return np.sqrt((data["x"]-c[0])**2 + (data["y"]-c[1])**2 + (data["z"]-c[2])**2)

        def _rdotPi_x(field, data):
            return data['x']*(data['pressure'] + data['density']*data['velocity_x']**2) +\
                   data['y']*(data['density']*data['velocity_y']*data['velocity_x']) +\
                   data['z']*(data['density']*data['velocity_z']*data['velocity_x'])

        def _rdotPi_y(field, data):
            return data['y']*(data['pressure'] + data['density']*data['velocity_y']**2) +\
                   data['x']*(data['density']*data['velocity_x']*data['velocity_y']) +\
                   data['z']*(data['density']*data['velocity_z']*data['velocity_y'])

        def _rdotPi_z(field, data):
            return data['z']*(data['pressure'] + data['density']*data['velocity_z']**2) +\
                   data['x']*(data['density']*data['velocity_x']*data['velocity_z']) +\
                   data['y']*(data['density']*data['velocity_y']*data['velocity_z'])

        # Note the four*pi difference (need to divide by 4pi later)
        def _rdotTM_x(field, data):
            return data['x']*0.5*(data['magnetic_field_x']**2 - data['magnetic_field_y']**2 -\
                                  data['magnetic_field_z']**2) +\
                   data['y']*(data['magnetic_field_x']*data['magnetic_field_y']) +\
                   data['z']*(data['magnetic_field_x']*data['magnetic_field_z'])

        def _rdotTM_y(field, data):
            return data['y']*0.5*(data['magnetic_field_y']**2 - data['magnetic_field_z']**2 -\
                                  data['magnetic_field_x']**2) +\
                   data['x']*(data['magnetic_field_x']*data['magnetic_field_y']) +\
                   data['z']*(data['magnetic_field_y']*data['magnetic_field_z'])

        def _rdotTM_z(field, data):
            return data['z']*0.5*(data['magnetic_field_z']**2 - data['magnetic_field_x']**2 -\
                                  data['magnetic_field_y']**2) +\
                   data['x']*(data['magnetic_field_x']*data['magnetic_field_z']) +\
                   data['y']*(data['magnetic_field_y']*data['magnetic_field_z'])

        def _rhovrsq_x(field, data):
            return data['density']*data['velocity_x']*data['dist']**2

        def _rhovrsq_y(field, data):
            return data['density']*data['velocity_y']*data['dist']**2

        def _rhovrsq_z(field, data):
            return data['density']*data['velocity_z']*data['dist']**2

        def _rho_rsq(field, data):
            return data['density']*data['dist']**2

        ds.add_field("dist", function=_dist, units="pc", sampling_type='cell', force_override=True)
        ds.add_field("rdotPi_x", function=_rdotPi_x, units="erg/cm**2", sampling_type='cell', force_override=True)
        ds.add_field("rdotPi_y", function=_rdotPi_y, units="erg/cm**2", sampling_type='cell', force_override=True)
        ds.add_field("rdotPi_z", function=_rdotPi_z, units="erg/cm**2", sampling_type='cell', force_override=True)
        ds.add_field("rdotTM_x", function=_rdotTM_x, units="erg/cm**2", sampling_type='cell', force_override=True)
        ds.add_field("rdotTM_y", function=_rdotTM_y, units="erg/cm**2", sampling_type='cell', force_override=True)
        ds.add_field("rdotTM_z", function=_rdotTM_z, units="erg/cm**2", sampling_type='cell', force_override=True)
        ds.add_field("rhovrsq_x", function=_rhovrsq_x, units="g/s", sampling_type='cell', force_override=True)
        ds.add_field("rhovrsq_y", function=_rhovrsq_y, units="g/s", sampling_type='cell', force_override=True)
        ds.add_field("rhovrsq_z", function=_rhovrsq_z, units="g/s", sampling_type='cell', force_override=True)
        ds.add_field("rho_rsq", function=_rho_rsq, units="g*cm**-1", sampling_type='cell', force_override=True)
        ds.add_gradient_fields(("athena","gravitational_potential"))

        return ds

