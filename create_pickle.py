from __future__ import print_function

import numpy as np
import pickle
import glob,os

import astropy.constants as ac
import astropy.units as au
import pyathena as pa
from pyathena.utils import compare_files
from pyathena.set_units import *
from pyathena.plot_tools.plot_projection import plot_projection
from pyathena.plot_tools.plot_slices import slice2 as plot_slice
from pyathena.plot_tools.plot_slice_proj import plot_slice_proj as plot_slice_proj
from pyathena.plot_tools.set_aux import set_aux
from pyathena.cooling import coolftn

coolftn = coolftn()
unit = set_units(muH=1.4271)

to_Myr = unit['time'].to('Myr').value
to_Pok = (unit['pressure']/ac.k_B).cgs.value
to_microG = unit['magnetic_field'].value
to_surf = (unit['density']*unit['length']).to('Msun/pc^2').value

data_axis={'x':2,'y':1,'z':0}
domain_axis={'x':0,'y':1,'z':2}
proj_axis={'z':('x','y'),'y':('x','z'),'x':('y','z')}

def get_scalars(ds):
    scal_fields=[]
    for f in ds.field_list:
        if f.startswith('specific_scalar'):
            scal_fields.append(f)

    return scal_fields

def create_surface_density(ds,surf_fname):
    '''
        specific function to create pickle file containing surface density
    '''
    
    time = ds.domain['time']*to_Myr
    dx=ds.domain['dx']

    surf_data={}
    le=ds.domain['left_edge']
    re=ds.domain['right_edge']
    pdata=ds.read_all_data('density')

    proj = pdata.mean(axis=0)
    proj *= ds.domain['Lx'][2]*to_surf
    bounds = np.array([le[0],re[0],le[1],re[1]])
    surf_data={'time':time,'data':proj,'bounds':bounds}
    pickle.dump(surf_data,open(surf_fname,'wb'),pickle.HIGHEST_PROTOCOL)
    
def create_projections(ds, fname, fields, weight_fields=dict(), aux=None,
                       force_recal=False, verbose=False):
    """
    Generic function to create pickle file containing projections of field along all axes
    
    Parameters
    ----------
       ds: AthenaDataset
       fname: string
          Name of pickle file to save data
       fields: list of strings
          List of field names to be projected
       weight_fields: dictionary
          Dictionary for weight fields to be multiplied to the projected field
       aux: dictionary
          aux dictionary
       force_recal: bool
          If True, override existing pickles
       verbose: bool
          If True, print verbose messages
    """
    
    time = ds.domain['time']*to_Myr
    dx = ds.domain['dx']

    le = ds.domain['left_edge']
    re = ds.domain['right_edge']

    if aux is None:
        aux = set_aux()
        
    for f in fields:
        fp = f + '_proj'
        if fp in aux and 'weight_field' in aux[fp]:
            weight_fields[f] = aux[fp]['weight_field']
        
    import copy
    field_to_proj = copy.copy(fields)
    # Check if pickle exists and remove existing fields from field_to_proj
    if os.path.isfile(fname) and not force_recal:
        proj_data = pickle.load(open(fname, 'rb'))
        existing_fields = proj_data['z'].keys()
        for f in existing_fields:
            if f in field_to_proj:
                #print('Pickle for {} exists.'.format(f))
                field_to_proj.remove(f)
    else:
        proj_data = {}
        proj_data['time'] = time
        for i, axis in enumerate(['x', 'y', 'z']):
            bounds = np.array([le[domain_axis[proj_axis[axis][0]]],
                               re[domain_axis[proj_axis[axis][0]]],
                               le[domain_axis[proj_axis[axis][1]]],
                               re[domain_axis[proj_axis[axis][1]]]])
            proj_data[axis] = {}
            proj_data[axis + 'extent'] = bounds

    if verbose:
        print('making projections...', end='')
    for f in field_to_proj:
        if verbose:
            print('{}...'.format(f),end='')
        pdata = ds.read_all_data(f)
        if isinstance(weight_fields,dict) and f in weight_fields:
            if weight_fields[f] == 'cell_volume':
                wdata = ds.domain['dx'].prod()*np.ones(pdata.shape)
            else:
                wdata = ds.read_all_data(weight_field[f])
            pdata *= wdata
        for i, axis in enumerate(['x', 'y', 'z']):
            dl_cgs = (ds.domain['dx'][domain_axis[axis]]*unit['length'].cgs).value
            proj = pdata.sum(axis=data_axis[axis])*dl_cgs
            if isinstance(weight_fields, dict) and f in weight_fields:
                wproj = wdata.sum(axis=data_axis[axis])*dl_cgs
                proj /= wproj

            bounds = np.array([le[domain_axis[proj_axis[axis][0]]],
                               re[domain_axis[proj_axis[axis][0]]],
                               le[domain_axis[proj_axis[axis][1]]],
                               re[domain_axis[proj_axis[axis][1]]]])

            try:
                proj *= aux[f]['proj_mul']
            except KeyError:
                #print('proj field {}: multiplication factor not available in aux.'.format(f))
                pass
                
            proj_data[axis][f] = proj
    if verbose:
        print('')
    
    pickle.dump(proj_data, open(fname, 'wb'), pickle.HIGHEST_PROTOCOL)
    return proj_data

def create_slices(ds,fname,fields,force_recal=False,factors={},verbose=False):
    '''
        generic function to create pickle file containing slices of fields
    
        fields: list of field names to be sliced
        
        factors: multiplication factors for unit conversion
    '''
 
    time = ds.domain['time']*to_Myr
    dx=ds.domain['dx']
    c=ds.domain['center']
    le=ds.domain['left_edge']
    re=ds.domain['right_edge']
    cidx=pa.cc_idx(ds.domain,ds.domain['center']).astype('int') 

    import copy
    field_to_slice = copy.copy(fields)
    if os.path.isfile(fname) and not force_recal:
        slc_data = pickle.load(open(fname,'rb'))
        existing_fields = slc_data['z'].keys()
        for f in existing_fields:
            if f in field_to_slice: 
                #print('Pickle for {} exists'.format(f))
                field_to_slice.remove(f)
    else:
        slc_data={}
        slc_data['time']=time
       
        for i,axis in enumerate(['x','y','z']):
            bounds = np.array([le[domain_axis[proj_axis[axis][0]]],
                               re[domain_axis[proj_axis[axis][0]]],
                               le[domain_axis[proj_axis[axis][1]]],
                               re[domain_axis[proj_axis[axis][1]]]])
            slc_data[axis] = {}
            slc_data[axis+'extent'] = bounds

    if verbose:
        print('making slices...', end='')
    for f in field_to_slice:
        if verbose:
            print('{}...'.format(f), end='')
        if f is 'temperature':
            if 'xn' in ds.derived_field_list:
                pdata = ds.read_all_data('temperature')
            else:
                pdata = ds.read_all_data('T1')
        elif f is 'magnetic_field_strength':
            pdata = ds.read_all_data('magnetic_field')
        elif f is 'ram_pok_z':
            pdata = ds.read_all_data('kinetic_energy3')*2.0
        elif f is 'pok':
            pdata = ds.read_all_data('pressure')
        elif f is 'velocity_z':
            pdata = ds.read_all_data('velocity3')
        elif f is 'mag_pok':
            pdata = ds.read_all_data('magnetic_pressure')
        elif f is 'nH':
            pdata = ds.read_all_data('density')
        else:
            pdata = ds.read_all_data(f)

        for i, axis in enumerate(['x','y','z']):
            if f == 'temperature' and not 'xn' in ds.derived_field_list:
                slc = coolftn.get_temp(pdata.take(cidx[i],axis=2-i))
            elif f is 'magnetic_field_strength':
                slc = np.sqrt((pdata.take(cidx[i],axis=2-i)**2).sum(axis=-1))
            else:
                slc = pdata.take(cidx[i],axis=2-i)

            if f in factors:
                slc_data[axis][f] = slc*factors[f]
            else:
                slc_data[axis][f] = slc
    if verbose:
        print('')
        
    pickle.dump(slc_data, open(fname, 'wb'), pickle.HIGHEST_PROTOCOL)
    return slc_data

def create_all_pickles_mhd(force_recal=False, force_redraw=False, verbose=True, **kwargs):
    """
    Original create_all_pickles used to extract and draw gas surface density and slices
    """
    dir = kwargs['base_directory']+kwargs['directory']
    fname=glob.glob(dir+'id0/'+kwargs['id']+'.????.vtk')
    fname.sort()

    if kwargs['range'] != '':
        sp=kwargs['range'].split(',')
        start = eval(sp[0])
        end = eval(sp[1])
        fskip = eval(sp[2])
    else:
        start = 0
        end = len(fname)
        fskip = 1
    fname=fname[start:end:fskip]

    #ngrids=len(glob.glob(dir+'id*/'+kwargs['id']+'*'+fname[0][-8:]))

    ds=pa.AthenaDataSet(fname[0])
    mhd='magnetic_field' in ds.field_list
    cooling='pressure' in ds.field_list

    Omega=kwargs['rotation']
    rotation=kwargs['rotation'] != 0.
    if verbose:
        print("MHD:", mhd)
        print("cooling:", cooling)
        print("rotation:", rotation, Omega)

    slc_fields=['nH','pok','temperature','velocity_z','ram_pok_z']
    fields_to_draw=['star_particles','nH','temperature','pok','velocity_z']
    if mhd:
        slc_fields.append('magnetic_field_strength')
        slc_fields.append('mag_pok')
        fields_to_draw.append('magnetic_field_strength')
    mul_factors={'pok':to_Pok,'magnetic_field_strength':to_microG,'mag_pok':to_Pok,'ram_pok_z':to_Pok}

    scal_fields=get_scalars(ds)
    slc_fields+=scal_fields

    if not os.path.isdir(dir+'slice/'): os.mkdir(dir+'slice/')
    if not os.path.isdir(dir+'surf/'): os.mkdir(dir+'surf/')

    for i,f in enumerate(fname):
        slcfname=dir+'slice/'+kwargs['id']+f[-9:-4]+'.slice.p'
        surfname=dir+'surf/'+kwargs['id']+f[-9:-4]+'.surf.p'

        tasks={'slice':(not compare_files(f,slcfname)) or force_recal,
               'surf':(not compare_files(f,surfname)) or force_recal,
        }

        do_task=(tasks['slice'] or tasks['surf'])
        
        if verbose: 
            print('file number: {} -- Tasks to be done ['.format(i),end='')
            for k in tasks: print('{}:{} '.format(k,tasks[k]),end='')
            print(']')
        if do_task:
            ds = pa.AthenaDataSet(f)
            if tasks['surf']: create_projection(ds,surfname,conversion={'z':ds.domain['Lx'][2]*to_surf})
            if tasks['slice']: create_slices(ds,slcfname,slc_fields,factors=mul_factors,force_recal=force_recal)

    aux=set_aux(kwargs['id'])

    for i,f in enumerate(fname):
        slcfname=dir+'slice/'+kwargs['id']+f[-9:-4]+'.slice.p'
        surfname=dir+'surf/'+kwargs['id']+f[-9:-4]+'.surf.p'

        starpardir='id0/'
        if os.path.isdir(dir+'starpar/'): starpardir='starpar/'
        starfname=dir+starpardir+kwargs['id']+f[-9:-4]+'.starpar.vtk'

        tasks={'slice':(not compare_files(f,slcfname+'ng')) or force_redraw,
               'surf':(not compare_files(f,surfname+'ng')) or force_redraw,
        }
        do_task=(tasks['slice'] and tasks['surf'])
        if verbose:
            print('file number: {} -- Tasks to be done ['.format(i),end='')
            for k in tasks: print('{}:{} '.format(k,tasks[k]),end='')
            print(']')
        if tasks['surf']:
            plot_projection(surfname,starfname,runaway=True,aux=aux['surface_density'])
        if tasks['slice']:
            plot_slice(slcfname,starfname,fields_to_draw,aux=aux)

def create_all_pickles(
        datadir, problem_id,
        nums=None,
        fields_slc=['nH', 'nHI', 'temperature', 'xn', 'ne', 'nesq', 'Erad0', 'Erad1'],
        fields_proj=['rho', 'xn', 'nesq'],
        fields_draw=['star_particles', 'rho_proj', 'xn_proj', 'nesq_proj',
                     'nH', 'temperature', 'xn', 'Erad0', 'Erad1'],
        force_recal=False, force_redraw=False, no_save=False,
        verbose=True, **plt_args):
    """
    --------------------------------------------------------------------------------
    Function to pickle slices and projections from AthenaDataset and draw snapshots.
    Set force_recal to True if additional fields need to be extracted.

    Parameters
    ----------
       datadir: string
          Base data directory
       problem_id: string
          Prefix for vtk files
       num: array of integers
          List of vtk output numbers. Search all vtk files in the directory if None.
       fields_slc: list of strings
          List of field names to be sliced
       fields_proj: list of strings
          List of field names to be projected
       fields_draw: list of strings
          List of field names to be drawn
       force_recal: bool
          If True, override existing pickles.
       force_redraw: bool
          If True, override existing figures.
       no_save: bool
          If True, returns a list of matplotlib figure objects instead of 
          saving them.
       verbose: bool
          Print verbose message
    
    Returns
    -------
       fig: figures
          Returns lists of figure if no_save is True.
    """
    
    aux = set_aux(problem_id)
    _plt_args = dict(zoom=1.0)
    _plt_args.update(**plt_args)
    
    fglob = os.path.join(datadir, problem_id + '.????.vtk')
    fname = sorted(glob.glob(fglob))
    if not fname:
        fglob = os.path.join(datadir, 'vtk', problem_id + '.????.vtk')
        fname = glob.glob(fglob)
    if not fname:
        fglob = os.path.join(datadir, 'id0', problem_id + '.????.vtk')
        fname = glob.glob(fglob)

    fname.sort()
    if not fname:
        print('No vtk files are found in {0:s}'.format(datadir))
        raise
    
    if nums is None:
        nums = [int(f[-8:-4]) for f in fname]
        if nums[0] == 0: # remove the zeroth snapshot
            start = 1
            del nums[0]
        else:
            start = 0
            
        end = len(fname)
        fskip = 1
        fname = fname[start:end:fskip]
    else:
        nums = np.atleast_1d(nums)
        fname = [fname[i] for i in nums]
        
    #ngrids = len(glob.glob(datadir+'id*/' + id + '*' + fname[0][-8:]))

    ds = pa.AthenaDataSet(fname[0])
    mhd = 'magnetic_field' in ds.field_list
    cooling = 'pressure' in ds.field_list

    print('[Create_pickle_all_rad]')
    print('- basedir:', datadir)
    print('- problem id:', problem_id)
    print('- vtk file num:', end=' ')
    for i in nums:
        print(i,end=' ')
    print('slc: {0:s}'.format(' '.join(fields_slc)))
    print('proj: {0:s}'.format(' '.join(fields_proj)))
    print('draw: {0:s}'.format(' '.join(fields_draw)))
    
    if mhd:
        slc_fields.append('magnetic_field_strength')
        slc_fields.append('mag_pok')
        draw_fields.append('magnetic_field_strength')
    mul_factors = {'pok':to_Pok,
                   'magnetic_field_strength':to_microG,
                   'mag_pok':to_Pok,
                   'ram_pok_z':to_Pok}

    if not os.path.isdir(os.path.join(datadir, 'slice')):
        os.mkdir(os.path.join(datadir, 'slice'))
    if not os.path.isdir(os.path.join(datadir, 'proj')):
        os.mkdir(os.path.join(datadir, 'proj'))

    print('')
    print('*** Extract slices and projections ***')
    print('- num: ',end='')
    for i, f in enumerate(fname):
        print('{}'.format(int(f.split('.')[-2])), end=' ')
        fname_slc = os.path.join(datadir, 'slice', problem_id + f[-9:-4] + '.slice.p')
        fname_proj = os.path.join(datadir, 'proj', problem_id + f[-9:-4] + '.proj.p')

        tasks=dict(slc=(not compare_files(f,fname_slc)) or force_recal,
                   proj=(not compare_files(f,fname_proj)) or force_recal)

        do_task = (tasks['slc'] or tasks['proj'])
        if do_task:
            ds = pa.AthenaDataSet(f)
            if tasks['slc']:
                create_slices(ds, fname_slc, fields_slc, factors=mul_factors,
                              force_recal=force_recal, verbose=verbose)
            if tasks['proj']:
                create_projections(ds, fname_proj, fields_proj, aux=aux,
                                   force_recal=force_recal, verbose=verbose)

    print('')
    print('*** Draw snapshots (zoom {0:.1f}) ***'.format(_plt_args['zoom']))
    print('num: ',end='')
    if no_save:
        force_redraw = True
        figs = []

    savdir = os.path.join(datadir, 'snapshots')
    if not os.path.isdir(savdir):
        os.mkdir(savdir)
    print('savdir:', savdir)
    for i,f in enumerate(fname):
        num = f.split('.')[-2]
        print('{}'.format(int(num)), end=' ')
        fname_slc = os.path.join(datadir, 'slice',
                                 problem_id + f[-9:-4] + '.slice.p')
        fname_proj = os.path.join(datadir, 'proj',
                                  problem_id + f[-9:-4] + '.proj.p')

        starpardir = 'id0'
        if os.path.isdir(os.path.join(datadir, 'starpar')):
            starpardir='starpar'
        fname_sp = os.path.join(datadir, starpardir,
                                problem_id + f[-9:-4] + '.starpar.vtk')

        savname = os.path.join(savdir, problem_id + '.' + num + '.slc_proj.png')
        if _plt_args['zoom'] == 1.0:
            savname = os.path.join(savdir, problem_id + '.' + num + '.slc_proj.png')
        else:
            # append zoom factor
            savname = os.path.join(savdir, problem_id + '.' + num + '.slc_proj-' + \
                                   'zoom{0:02d}'.format(int(10.0*_plt_args['zoom'])) + '.png')

        tasks = dict(slc_proj=(not compare_files(f, savname)) or force_redraw,
                     proj=(not compare_files(f, fname_proj+'ng')) or force_redraw)
        
        do_task = (tasks['slc_proj'] and tasks['proj'])
        if tasks['proj']:
            plot_projection(fname_proj, fname_sp, 'rho', runaway=True,
                            aux=aux['rho_proj'])
        if tasks['slc_proj']:
            if no_save:
                savname = None
                fig = plot_slice_proj(fname_slc, fname_proj, fname_sp, fields_draw,
                                      savname, aux=aux, **_plt_args)
                figs.append(fig)
            else:
                plot_slice_proj(fname_slc, fname_proj, fname_sp, fields_draw,
                                savname, aux=aux, **_plt_args)
            
    print('')
    print('*** Done! ***')

    if no_save:
        return tuple(figs)
