import pandas as pd
import os.path as osp

def read_bialy19(basedir='./Bialy19', metal='Z1', new=True):
    """Read Bialy+2019 Equilibrium Curves
    
    Parameters
    ----------
    basedir : str
        Directory where files are located
    metal : str
        Metallicity. 'Z1' or 'Z01'

    Cooling processes:
    - [CII]
    - [OI]
    - Ly-Alpha
    - recombination on dust
    - H2 dissociation
    - chemical reaction with H2 (charge transfer)
    - H2 rovibrational line emission
    - HD cooling
    - [CI]
    - dust-gas reactions

    Heating processes:
    - photoelectric heating
    - Ionization heating (by CR ionization)
    - H2 photodissociation heating
    - H2 formation heating
    - H2 FUV pumping heating
    - PdV work
    - dust-gas interactions
    """
    
    if new:
        basedir = osp.join(basedir, 'new')
    else:
        basedir = osp.join(basedir, 'old')
    
    fname_nTP = osp.join(basedir, 'n_T_P_{0:s}.txt'.format(metal))
    if new:
        fname_xe = osp.join(basedir, 'xi_{0:s}.txt'.format(metal))
    else:
        fname_xe = osp.join(basedir, 'xe_{0:s}.txt'.format(metal))
    fname_cool = osp.join(basedir, 'cool_{0:s}.txt'.format(metal))
    fname_heat = osp.join(basedir, 'heat_{0:s}.txt'.format(metal))
    
    BnTP = pd.read_csv(fname_nTP, sep='\t', index_col=False, names=['nH','T','pok'])
    if new:
        Be = pd.read_csv(fname_xe, sep='\t', index_col=False, names=['xe','xH+','xC+'])
    else:
        Be = pd.read_csv(fname_xe, sep='\t', index_col=False, names=['xe'])
        
    Bc = pd.read_csv(fname_cool, sep='\t', index_col=False, 
                     names=['coolCII','coolOI','coolLya','coolH2diss',
                            'coolH2chem','coolH2rovib','coolHD','coolCI','cooldust'])
    Bh = pd.read_csv(fname_heat, sep='\t', index_col=False,
                     names=['heatPE','heatCR','heatH2diss','heatH2form',
                            'heatH2pump','heatpdV','heatdust'])

    B = pd.concat([BnTP,Be,Bc,Bh],axis=1)

    # Compute total cooling and heating
    B['cooltot'] = 0.0
    B['heattot'] = 0.0
    for col in B.columns:
        if col.startswith('cool') and col != 'cooltot':
            B[col] /= B['nH']
            B['cooltot'] += B[col]
        if col.startswith('heat') and col != 'heattot':
            B[col] /= B['nH']
            B['heattot'] += B[col]

    return B
