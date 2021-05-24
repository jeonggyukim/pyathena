#!/usr/bin/env python

import glob
import os
import os.path as osp
import argparse
from mpi4py import MPI
from pyathena.util.split_container import split_container

def compare_files(source, output):
    '''compare source and desination files and return True if source exist and older
    '''
    if os.path.isfile(source):
        smtime=os.path.getmtime(source)
    else:
        return True
    if os.path.isfile(output):
        omtime=os.path.getmtime(output)
        if omtime < smtime:
            return False
        else:
            return True
    else:
        return False

basedir_orig_def = "/perseus/scratch/gpfs/changgoo/TIGRESS-NCR/R8s_2pc_NCR"
basedir_orig_def = "/perseus/scratch/gpfs/changgoo/TIGRESS-NCR/R8s_8pc_NCR_Z01"
basedir_new_def = "/tigress/changgoo/TIGRESS-NCR/R8s_8pc_NCR_Z01"
join_vtk_script = "./vtk/join.sh"

join_vtk_def = False
join_vtk_suffix_def = False
sync_rst_def = False

parser = argparse.ArgumentParser(
    description='''Move tigress simulation output files from gpfsto tigress
using rsync. To move vtk files, use vtk/join_vtk.sh script''')

parser.add_argument('-i', '--basedir_orig', type=str,
                    default=basedir_orig_def,
                    help='original basedir')
parser.add_argument('-o', '--basedir_new', type=str,
                    default=basedir_new_def,
                    help='new basedir')
parser.add_argument('-j', '--join_vtk',
                    action='store_true', default=join_vtk_def,
                    help='Toggle to (3d) join vtk files')
parser.add_argument('-s', '--join_vtk_suffix',
                    action='store_true', default=join_vtk_suffix_def,
                    help='Toggle to join (2d) vtk files that have suffix')
parser.add_argument('-r', '--sync_rst',
                    action='store_true', default=sync_rst_def,
                    help='Toggle to sync restart files')

args = vars(parser.parse_args())
locals().update(args)

basedir_orig_id0 = osp.join(basedir_orig, 'id0', '') # add trailing slash

COMM = MPI.COMM_WORLD

if COMM.rank == 0:
    print('basedir_orig: ', basedir_orig)
    print('basedir_new: ', basedir_new)

    if not osp.isdir(basedir_orig):
        raise IOError('basedir_orig does not exist: ', basedir_orig)

    if osp.isdir(basedir_new):
        print('New basedir {0:s} exists.'.format(basedir_new))
    else:
        print('Create new basedir {0:s}'.format(basedir_new))
        os.makedirs(basedir_new)


basedir_new_sub = dict()
basedir_new_sub['hst'] = osp.join(basedir_new, 'hst')
basedir_new_sub['starpar'] = osp.join(basedir_new, 'starpar')
basedir_new_sub['rst'] = osp.join(basedir_new, 'rst')
basedir_new_sub['vtk'] = osp.join(basedir_new, 'vtk')
if glob.glob(osp.join(basedir_orig, 'id0', '*.zprof')):
    basedir_new_sub['zprof'] = osp.join(basedir_new, 'zprof')

if COMM.rank == 0:
    for k,d in basedir_new_sub.items():
        if not osp.isdir(d):
            os.makedirs(d)
            print('Create directory for {0:s}: {1:s}'.format(k,d))

COMM.barrier()

rsync_id0 = 'rsync -av {0:s} {1:s}'.format(basedir_orig_id0, basedir_new)

rsync_hst = 'rsync -av --include="*.sn" --include="*.hst" --exclude="*" {0:s} {1:s}'.\
                                  format(basedir_orig_id0, basedir_new_sub['hst'])
rsync_star = 'rsync -av --include="*.starpar.vtk" --include="*.star" --exclude="*" {0:s} {1:s}'.\
                                   format(basedir_orig_id0, basedir_new_sub['starpar'])
rsync_rst = 'rsync -av --include="*.rst" --exclude="*" {0:s}/id*/ {1:s}'.\
                                   format(basedir_orig, basedir_new_sub['rst'])
if 'zprof' in basedir_new_sub.keys():
    rsync_zprof = 'rsync -av --include="*.zprof" --exclude="*" {0:s} {1:s}'.\
                                        format(basedir_orig_id0, basedir_new_sub['zprof'])

# rsync_misc = 'rsync -av --include="snapshots" --include="prj*" --include="slc*"  --include="athinput*" --include="athena*" --include="radps_postproc*" --include="tigress*" --include="*.txt" ' + '--exclude="*" {0:s} {1:s}'.format(osp.join(basedir_orig, ''), basedir_new)
rsync_misc = 'rsync -av --exclude="id*" {0:s} {1:s}'.format(osp.join(basedir_orig, ''), basedir_new)

if join_vtk or join_vtk_suffix:
    if COMM.rank == 0:
        print('##################')
        print('# join vtk files')
        print('##################')

    # Find all vtk files in id0 directory except for starpar.vtk
    nums = dict()
    nums['vtk'] = []
    suffix = []
    fvtk = glob.glob(osp.join(basedir_orig, 'id0', '*.vtk'))
    for f in fvtk:
        ff = osp.basename(f).split('.')
        prefix = ff[0] # problem_id
        try:
            nums['vtk'].append(int(ff[-2]))
        except ValueError:
            suffix.append(ff[-2])

    nums['vtk'] = sorted(nums['vtk'])

    # Find all vtk with suffix
    suffix = list(set(suffix))
    suffix.remove('starpar')
    for s in suffix:
        nums[s] = []
        fvtk = glob.glob(osp.join(basedir_orig, 'id0', f'*.{s}.vtk'))
        for f in fvtk:
            ff = osp.basename(f).split('.')
            nums[s].append(int(ff[-3]))

        nums[s] = sorted(nums[s])

    # Range
    join_vtk_command = dict()
    if join_vtk_suffix:
        for s in suffix:
            num_min = nums[s][0]
            if osp.exists(osp.join(basedir_new,s)):
                for num in nums[s]:
                    if osp.exists(osp.join(basedir_new,s,
                                           '{0:s}.{1:04d}.{2:s}.vtk'.format(prefix,num,s))):
                        num_min = num
            r = r'{0:d}:{1:d}'.format(num_min,nums[s][-1])
            join_vtk_command[s] = '{0:s} -r {1:s} -i {2:s} -o {3:s} -s {4:s} -C'.format(
                join_vtk_script,r,basedir_orig,osp.join(basedir_new,s),s)
        # os.system(join_vtk_command)
        for k,v in join_vtk_command.items():
            if k != 'vtk':
                os.system(v)

    num_min = nums['vtk'][0]
    is_new = False
    inum = 0
    new_nums = []
    for inum,num in enumerate(nums['vtk']):
        orig_vtk=osp.join(basedir_orig,'id0','{0:s}.{1:04d}.vtk'.format(prefix,num))
        target_vtk=osp.join(basedir_new,'vtk','{0:s}.{1:04d}.vtk'.format(prefix,num))
        if compare_files(orig_vtk,target_vtk):
            num_min = num
        else:
            is_new = True
            new_nums.append(num)

    if is_new:
        if COMM.rank == 0:
            print('new nums', new_nums)
            new_nums = split_container(new_nums, COMM.size)
        else:
            new_nums = None

        mynums = COMM.scatter(new_nums, root=0)
        print('[rank, mynums]:', COMM.rank, mynums)

        mymin = mynums[0]
        mymax = mynums[-1]
        if len(mynums) > 1: mystride = mynums[1]-mynums[0]
        else: mystride = 1
        print(COMM.rank, mynums, mymin, mymax, mystride)
        r = r'{0:d}:{1:d}:{2:d}'.format(mymin,mymax,mystride)
        join_vtk_command['vtk'] = '{0:s} -r {1:s} -i {2:s} -o {3:s} -C'.format(
            join_vtk_script,r,basedir_orig,osp.join(basedir_new,'vtk'))
        print(join_vtk_command['vtk'])
        os.system(join_vtk_command['vtk'])

COMM.barrier()

if COMM.rank == 0:
    #vtkfiles=os.listdir(osp.join(basedir_new,'vtk'))
    #vtkfiles.sort()
    #print(vtkfiles)

    if sync_rst:
        print('##################')
        print('# rsync rst files')
        print('##################')
        os.system(rsync_rst)

    commands = [rsync_hst, rsync_star, rsync_misc]
    if 'zprof' in basedir_new_sub.keys():
        commands.append(rsync_zprof)

    for c in commands:
        print(c)
        os.system(c)
