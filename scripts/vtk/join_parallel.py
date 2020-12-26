#!/usr/bin/env python

from __future__ import print_function
import os, argparse, subprocess, socket
try:
    from mpi4py import MPI
except:
    pass

def split_container(container, count):
    """
    Original source:
    https://gist.github.com/krischer/2c7b95beed642248487a

    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

def eval_range(string, end_def=2001):
    """
    Returns range(start, end, step) from input string
    'start:end:step' (see examples below)

    Examples:
    'b' - range(b,b+1)
    'a:b' - range(a,b)
    'a:b:c' - range(a,b,c)
    ':b:c' - range(0,b,c)
    '::c' - range(0,end,c)
    'a::c' - range(a,end,c)
    """
    
    r = None
    if string.count(':') == 2:
        s = string.split(':')
        if s[0] and s[1] and s[2]:        # full info given
            r = range(eval(s[0]), eval(s[1])+1, eval(s[2]))
        elif not s[0] and s[1] and s[2]:   # start missing
            r = range(0, eval(s[1])+1, eval(s[2]))
        elif not s[0] and not s[1] and s[2]: # only step size given
            r = range(0, end_def, eval(s[2]))
        elif s[0] and s[1] and not s[2]:   # step missing
            r = range(eval(s[0]),eval(s[1])+1)
        elif s[0] and not s[1] and s[2]:   # end missing
            r = range(eval(s[0]),end_def,eval(s[2]))
    elif string.count(':') == 1:
        s = string.split(':')
        if s[0] and s[1]:
            r = range(eval(s[0]), eval(s[1])+1)
    elif string.count(':') == 0:
        r = range(eval(string), eval(string)+1)
        
    return r

def main(join, **kwargs):

    if kwargs['range'] is None:
        raise ValueError("range should be specified.")
    
    try:
        COMM=MPI.COMM_WORLD
        if COMM.rank == 0:
            if kwargs['range'] is None:
                raise ValueError('Range error:',kwargs['range'])
            else:
                steps_all=eval_range(kwargs['range'])
            steps = split_container(steps_all,COMM.size)
        else:
            steps = None
        # Scatter steps across cores
        mysteps = COMM.scatter(steps, root=0)
        print('rank:', COMM.rank)
        print('mysteps:', mysteps)
    except:
        mysteps = eval_range(kwargs['range'])
        print('mysteps (no mpi):', mysteps)

    for s in mysteps:
        cmd=[join,
             '-i', kwargs['indir'],
             '-o', kwargs['outdir'],
             '-b', kwargs['basename'],
             '-r', '{0:d}:{0:d}'.format(s)]

        #print "rank:",COMM.rank," "," ".join(cmd)
        p = subprocess.Popen(cmd,stdout=subprocess.PIPE)
        for line in p.stdout:
            print(line, end='')
            p.wait()

        #print p.returncode
        
    return
   
if __name__ == '__main__':

    hostname = socket.gethostname()
    print('Hostname: ',hostname)

    # C compiled join binary file
    join = os.path.join(os.path.dirname(__file__),
                        'join.sh')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str, default='',
                        help='input dir Directory where vtk files are located')
    parser.add_argument('-o', '--outdir', type=str, default='',
                        help='Directory where joined vtk files will be stored')
    parser.add_argument('-b', '--basename', type=str, default='',
                        help='Basename of output files, e.g., BASENAME.xxxx.vtk')
    parser.add_argument('-r', '--range', dest='range', default=None,
                        help='range, start:end:stride')
    args = parser.parse_args()

    main(join, **vars(args))
