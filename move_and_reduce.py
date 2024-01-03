import pyathena as pa
import os
import time
import subprocess

def move_and_reduce(src,todir = '/tigress/changgoo/TIGRESS-NCR-lowZ/', ask=True):
    dest = os.path.join(todir,os.path.basename(src))
    print('Move files from {} to {}'.format(src,dest))
    subprocess.call(['rsync','-Ca','--info=progress2','-n',src,todir])
    subprocess.call(['checkquota'])
    if ask:
        dotask = input('  want to run rsync? [y/n]:')
    else:
        dotask = 'y'
    if dotask == 'y':
        subprocess.call(['rsync','-Ca','--info=progress2',src,todir])
    else:
        print(' Aborted!')
        return
    print('   cleaning up source folder')

    def reduce(src,dest,vtkskip=10,dry_run=True):
        s = pa.LoadSimTIGRESSNCR(src)
        nrst_remove = 0
        if len(s.nums_rst) > 1:
            print('deleting ...')
            for num in s.nums_rst[:-1]:
                cmd = ['rm','-rd',os.path.join(s.basedir,'rst','{:04d}'.format(num))]
                print('    {}'.format(cmd[-1]))
                if dry_run:
                    nrst_remove += 1
                else:
                    subprocess.call(cmd)
        nvtk_remove=0
        for num in s.nums:
            fvtk = s._get_fvtk('vtk_tar',num)
            if not (num % vtkskip):
                print('    {} kept'.format(os.path.basename(fvtk)))
                continue
            if dry_run:
                nvtk_remove += 1
            else:
                os.remove(fvtk)
        if dry_run:
            print('   {} rst {} vtk_tar -->'.format(len(s.nums_rst), len(s.nums)), end=' ')
            print('{} rst {} vtk_tar'.format(len(s.nums_rst) - nrst_remove, len(s.nums) - nvtk_remove))
        else:
            s = pa.LoadSimTIGRESSNCR(dest)
            print('   {} rst {} vtk_tar -->'.format(len(s.nums_rst), len(s.nums)), end=' ')
            s = pa.LoadSimTIGRESSNCR(src)
            print('{} rst {} vtk_tar'.format(len(s.nums_rst), len(s.nums)))

    reduce(src,dest,dry_run=True)
    if ask:
        dotask = input('  want to clean up? [y/n]:')
    else:
        dotask = 'y'
    if dotask == 'y':
        reduce(src,dest,dry_run=False)
    else:
        print(' cleaning is skipped!')
        return

if __name__ == '__main__':
    src = '/scratch/gpfs/changgoo/TIGRESS-NCR/LGR8_8pc_NCR_S05.full.b10.v3.iCR5.Zg0.1.Zd0.1.xy8192.eps0.0'
    move_and_reduce(src,ask=False)
