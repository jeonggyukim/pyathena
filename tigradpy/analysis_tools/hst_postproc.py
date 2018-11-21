import os, glob

from ..io.read_hst import read_hst
from ...vtk_reader import AthenaDataSet
from ..util.units import units

u = units()

def read_hst_postproc(fname, ds=None, nfreq=2, force_override=False):
    """
    Read hst and convert quantities to convenient units
    """
    hst = read_hst(fname, force_override=force_override)
    if ds is None:
        try:
            #print('[read_hst_postproc]: Cannot find vtk file '.foramt(fvtk))
            datadir = os.path.dirname(fname)
            problem_id = os.path.basename(fname[:-4])
            fvtk = sorted(glob.glob(os.path.join(datadir, problem_id + '.????.vtk')))
            #print('[read_hst_postproc]: Read vtk file ',fvtk[0])
            ds = AthenaDataSet(fvtk[0])
        except IOError:
            pass
        
    # volume of resolution element
    dvol = ds.domain['dx'].prod()
    # total volume of domain
    vol = ds.domain['Lx'].prod()
    # delete the first row
    hst.drop(hst.index[:1], inplace=True)
    
    hst['time'] *= u.Myr # time in Myr
    hst['mass'] *= vol*u.Msun # total gas mass in Msun
    hst['scalar3'] *= vol*u.Msun # neutral gas mass in Msun 
    hst['Mion_coll'] *= vol*u.Msun # (coll only before ray tracing) ionized gas mass in Msun
    hst['Mion'] *= vol*u.Msun # (coll + ionrad) ionized gas mass in Msun
    hst['Qirec'] *= vol*(u.lunit**3).cgs # recombination rate in cgs units
    hst['Qiphot'] *= vol*(u.lunit**3).cgs # photoionization rate in cgs units
    hst['Qicoll'] *= vol*(u.lunit**3).cgs # collisional ionization rate in cgs units
    hst['Qidust'] *= vol*(u.lunit**3).cgs # collisional ionization rate in cgs units
    
    for f in range(nfreq):
        # Total luminosity in Lsun
        hst['Ltot_cl{:d}'.format(f)] *= vol*u.Lsun
        hst['Ltot_ru{:d}'.format(f)] *= vol*u.Lsun
        hst['Ltot{:d}'.format(f)] = \
            hst['Ltot_cl{:d}'.format(f)] + hst['Ltot_ru{:d}'.format(f)]
        # Total luminosity included in Lsun
        hst['L_cl{:d}'.format(f)] *= vol*u.Lsun
        hst['L_ru{:d}'.format(f)] *= vol*u.Lsun
        hst['L{:d}'.format(f)] = \
            hst['L_cl{:d}'.format(f)] + hst['L_ru{:d}'.format(f)]
        # Luminosity that escaped boundary in Lsun
        hst['Lesc{:d}'.format(f)] *= vol*u.Lsun
        # Luminosity lost due to dmax
        hst['Llost{:d}'.format(f)] *= vol*u.Lsun
    
    #print(hst['L_cl0'],hst['Lesc0'])
    ##########################
    # With ionizing radiation
    hnu0 = (18.0/u.eV) # mean ionizing photon energy in code units
    #hnu1 = (10.0/u.eV) # mean ionizing photon energy in code units (arbitrary)
    # total ionizing photon rate
    hst['Qitot_cl'] = hst['Ltot_cl0']/u.Lsun/hnu0/u.s
    hst['Qitot_ru'] = hst['Ltot_ru0']/u.Lsun/hnu0/u.s
    hst['Qitot'] = hst['Qitot_ru'] + hst['Qitot_cl']

    # included as source
    hst['Qi_cl'] = hst['L_cl0']/u.Lsun/hnu0/u.s
    hst['Qi_ru'] = hst['L_ru0']/u.Lsun/hnu0/u.s
    hst['Qi'] = hst['Qi_ru'] + hst['Qi_cl']
    hst['Qiesc'] = hst['Lesc0']/u.Lsun/hnu0/u.s
    hst['Qilost'] = hst['Llost0']/u.Lsun/hnu0/u.s
    
    # midplane radiation energy density in cgs units
    hst['Erad0_mid'] *= u.punit 
    hst['Erad1_mid'] *= u.punit

    return hst
