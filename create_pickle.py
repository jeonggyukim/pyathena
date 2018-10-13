from __future__ import print_function

import numpy as np
import cPickle as pickle
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

coolftn=coolftn()
unit=set_units(muH=1.4271)

to_Myr=unit['time'].to('Myr').value
to_Pok=(unit['pressure']/ac.k_B).cgs.value
to_microG=unit['magnetic_field'].value
to_surf=(unit['density']*unit['length']).to('Msun/pc^2').value

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
    
def create_projection(ds, proj_fname, proj_fields, weight_fields=None, aux=None,
                      force_recal=False):
    """
    Generic function to create pickle file containing projections of field along all axes
    
    Parameters
    ----------
       ds: AthenaDataset
       proj_fname: string
          Name of pickle file to save data
       proj_fields: list of strings
          List of field names to be projected
       weight_fields: dictionary
          Dictionary for weight fields to be multiplied to the projected field
       aux: dictionary
          aux dictionary
       force_recal: bool
          If True, override existing pickles
    """
    
    time = ds.domain['time']*to_Myr
    dx = ds.domain['dx']

    le = ds.domain['left_edge']
    re = ds.domain['right_edge']

    if aux is None:
        aux = set_aux()
    
    import copy
    field_to_proj = copy.copy(proj_fields)
    if os.path.isfile(proj_fname) and not force_recal:
        proj_data = pickle.load(open(proj_fname, 'rb'))
        existing_fields = proj_data['z'].keys()
        for f in existing_fields:
            if f in field_to_proj:
                print('{} has already been pickled.'.format(f))
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
            proj_data[axis + 'extent'] = bounds/1.e3 # to kpc?

    print('making projections...', end='')
    for f in field_to_proj:
        print('{}...'.format(f),end='')
        pdata = ds.read_all_data(f)
        if isinstance(weight_fields,dict) and weight_fields.has_key('f'):
            wdata = ds.read_all_data(weight_field[f])
            pdata *= wdata
        for i, axis in enumerate(['x', 'y', 'z']):
            dl_cgs = (ds.domain['dx'][domain_axis[axis]]*unit['length'].cgs).value
            proj = pdata.sum(axis=data_axis[axis])*dl_cgs
            if isinstance(weight_fields, dict) and weight_fields.has_key(f):
                wproj = wdata.sum(axis=data_axis[axis])*dl_cgs
                proj /= wproj

            bounds = np.array([le[domain_axis[proj_axis[axis][0]]],
                               re[domain_axis[proj_axis[axis][0]]],
                               le[domain_axis[proj_axis[axis][1]]],
                               re[domain_axis[proj_axis[axis][1]]]])

            try:
                proj *=  aux[f]['proj_mul']
            except KeyError:
                pass
                #print('proj field {}: multiplication factor not available in aux.'.format(f))
                
            proj_data[axis][f] = proj
        
    print('')
    
    pickle.dump(proj_data, open(proj_fname, 'wb'), pickle.HIGHEST_PROTOCOL)
    return proj_data

def create_slices(ds,slcfname,slc_fields,force_recal=False,factors={}):
    '''
        generic function to create pickle file containing slices of fields
    
        slc_field: list of field names to be sliced
        
        factors: multiplication factors for unit conversion
    '''
 
    time = ds.domain['time']*to_Myr
    dx=ds.domain['dx']
    c=ds.domain['center']
    le=ds.domain['left_edge']
    re=ds.domain['right_edge']
    cidx=pa.cc_idx(ds.domain,ds.domain['center']).astype('int') 

    import copy
    field_to_slice = copy.copy(slc_fields)
    if os.path.isfile(slcfname) and not force_recal:
        slc_data = pickle.load(open(slcfname,'rb'))
        existing_fields = slc_data['z'].keys()
        for f in existing_fields:
            if f in field_to_slice: 
                print('{} is already there'.format(f))
                field_to_slice.remove(f)
    else:
        slc_data={}
        slc_data['time']=time
       
        for i,axis in enumerate(['x','y','z']):
            bounds = np.array([le[domain_axis[proj_axis[axis][0]]],re[domain_axis[proj_axis[axis][0]]],
                               le[domain_axis[proj_axis[axis][1]]],re[domain_axis[proj_axis[axis][1]]]])
            slc_data[axis]={}
            slc_data[axis+'extent']=bounds/1.e3

    print('making slices...',end='')
    for f in field_to_slice:
        print('{}...'.format(f),end='')
        if f is 'temperature':
            if 'xn' in ds.derived_field_list:
                pdata=ds.read_all_data('temperature')
            else:
                pdata=ds.read_all_data('T1')
        elif f is 'magnetic_field_strength':
            pdata=ds.read_all_data('magnetic_field')
        elif f is 'ram_pok_z':
            pdata=ds.read_all_data('kinetic_energy3')*2.0
        elif f is 'pok':
            pdata=ds.read_all_data('pressure')
        elif f is 'velocity_z':
            pdata=ds.read_all_data('velocity3')
        elif f is 'mag_pok':
            pdata=ds.read_all_data('magnetic_pressure')
        elif f is 'nH':
            pdata=ds.read_all_data('density')
        else:
            pdata=ds.read_all_data(f)

        for i,axis in enumerate(['x','y','z']):
            if f == 'temperature' and not 'xn' in ds.derived_field_list:
                slc=coolftn.get_temp(pdata.take(cidx[i],axis=2-i))
            elif f is 'magnetic_field_strength':
                slc=np.sqrt((pdata.take(cidx[i],axis=2-i)**2).sum(axis=-1))
            else:
                slc=pdata.take(cidx[i],axis=2-i)

            if f in factors:
                slc_data[axis][f] = slc * factors[f]
            else:
                slc_data[axis][f] = slc

    print('')
        
    pickle.dump(slc_data,open(slcfname,'wb'),pickle.HIGHEST_PROTOCOL)
    return slc_data

def create_all_pickles_mhd(force_recal=False, force_redraw=False, verbose=True, **kwargs):
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

def create_all_pickles_rad(
        datadir, problem_id,
        nums=None,
        slc_fields=['nH','nHI','temperature','xn','nesq','Erad0','Erad1'],
        proj_fields=['density','nesq'],
        draw_fields=['star_particles','nH','temperature','xn','nesq','Erad0'],
        force_recal=False, force_redraw=False, verbose=True, **kwargs):
    """
    Pickle slices and projections from dataset and draw snapshots

    Parameters
    ----------
       datadir: string
          Base data directory. ex) /tigress/changgoo/R8_8pc_newacc
       problem_id: string
          Prefix used for vtk files
       slc_fields: list of strings
          List of field names to be sliced
       draw_fields: list of strings
          List of field names to be drawn
       force_recal: bool
          If True, override existing pickles
       verbose: bool
          Print verbose message
    """
    
    dir = datadir
    id = problem_id
    aux = set_aux(id)

    
    fglob = os.path.join(dir, id + '.????.vtk')
    fname = glob.glob(fglob)
    if fname is None:
        fglob = os.path.join(dir, 'id0', id + '.????.vtk')
        fname = glob.glob(fglob)
    
    fname.sort()
    if fname is None:
        print('No vtk files are found in {}'.format(dir))
    
    if nums is not None:
        sp = nums.split(':')
        start = eval(sp[0])
        end = eval(sp[1])
        fskip = eval(sp[2])
    else:
        start = 1
        end = len(fname)
        fskip = 1

    fname = fname[start:end:fskip]
    #ngrids = len(glob.glob(dir+'id*/' + id + '*' + fname[0][-8:]))

    ds = pa.AthenaDataSet(fname[0])
    mhd = 'magnetic_field' in ds.field_list
    cooling = 'pressure' in ds.field_list

    if verbose:
        print('[Create_pickle_all_rad]')
        print('basedir:',dir)
        print('problem id:',id)
        print('vtk file numbers:',end=' ')
        for i in range(start,end,fskip):
            print(i,end=' ')
        print('')
        print("Magnetic fields:", mhd)
        print("cooling:", cooling)

    if mhd:
        slc_fields.append('magnetic_field_strength')
        slc_fields.append('mag_pok')
        draw_fields.append('magnetic_field_strength')

    mul_factors = {'pok':to_Pok,
                   'magnetic_field_strength':to_microG,
                   'mag_pok':to_Pok,
                   'ram_pok_z':to_Pok}

    ## Do not add specific_scalars
    #scal_fields=get_scalars(ds)
    #slc_fields+=scal_fields

    if not os.path.isdir(os.path.join(dir, 'slice')):
        os.mkdir(os.path.join(dir, 'slice'))
    if not os.path.isdir(os.path.join(dir, 'proj')):
        os.mkdir(os.path.join(dir, 'proj'))

    if verbose:
        print('')
        print('*** Extract slices and projections ***')
        
    for i, f in enumerate(fname):
        slcfname = os.path.join(dir,'slice',id + f[-9:-4] + '.slice.p')
        projfname = os.path.join(dir,'proj',id + f[-9:-4] + '.proj.p')

        tasks=dict(slc=(not compare_files(f,slcfname)) or force_recal,
                   proj=(not compare_files(f,projfname)) or force_recal)

        do_task = (tasks['slc'] or tasks['proj'])
        
        if verbose:
            print('num: {} -- Tasks ['.format(i), end='')
            for k in tasks: print('{}:{} '.format(k,tasks[k]),end='')
            print(']')
        if do_task:
            ds = pa.AthenaDataSet(f)
            if tasks['proj']:
                create_projection(ds, projfname, proj_fields, aux=aux)
            if tasks['slc']:
                create_slices(ds, slcfname, slc_fields, factors=mul_factors,
                              force_recal=force_recal)

    if verbose:
        print('')
        print('*** Draw snapshots ***')
    
    for i,f in enumerate(fname):
        slcfname=os.path.join(dir,'slice',id+f[-9:-4]+'.slice.p')
        projfname=os.path.join(dir,'proj',id+f[-9:-4]+'.proj.p')

        starpardir='id0'
        if os.path.isdir(os.path.join(dir,'starpar')):
            starpardir='starpar/'
        starfname=os.path.join(dir,starpardir,id+f[-9:-4]+'.starpar.vtk')

        tasks=dict(slc=(not compare_files(f,slcfname+'ng')) or force_redraw,
                   proj=(not compare_files(f,projfname+'ng')) or force_redraw)
        
        do_task=(tasks['slc'] and tasks['proj'])
        if verbose:
            print('file num: {} -- Tasks ['.format(i),end='')
            for k in tasks: print('{}:{} '.format(k,tasks[k]),end='')
            print(']')
        if tasks['proj']:
            plot_projection(projfname,starfname,'rho',runaway=True,
                            aux=aux['surface_density'])
        if tasks['slc']:
            #plot_slice(slcfname,starfname,draw_fields,aux=aux)
            plot_slice_proj(slcfname,projfname,starfname,draw_fields,aux=aux)

    if verbose:
        print('')
        print('*** Done! ***')
