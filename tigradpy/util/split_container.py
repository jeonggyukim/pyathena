#!/usr/bin/env python

from mpi4py import MPI
import numpy as np

def split_container(container, count):
    """
    Original source: https://gist.github.com/krischer/2c7b95beed642248487a
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

def split_N(COMM,N):
    """
    Distribute N consecutive things (rows of a matrix, blocks of a 1D array)
    as evenly as possible over a given COMMunicator.
    Uneven workload (differs by 1 at most) is on the initial ranks.

    Parameters
    ----------
    COMM : MPI COMMunicator
    N :  int
        Total number of things to be distributed.

    Returns
    -------
    rstart : index of first local row
    rend : 1 + index of last row
    """

    P      = COMM.size
    rank   = COMM.rank
    rstart = 0
    rend   = 0
    if P >= N:
        if rank < N:
            rstart = rank
            rend   = rank + 1
    else:
        n = N/P
        remainder = N%P
        rstart    = n * rank
        rend      = n * (rank+1)
        if remainder:
            if rank >= remainder:
                rstart += remainder
                rend   += remainder
            else: 
                rstart += rank
                rend   += rank + 1
    return rstart, rend

if __name__ == '__main__':

    # Simple tests
    COMM = MPI.COMM_WORLD
    size = COMM.Get_size()
    rank = COMM.Get_rank()

    print('size, rank', size, rank)
    print(split_N(COMM, 100))

    # Collect whatever has to be done in a list. Here we'll just collect a list of
    # numbers. Only the first rank has to do this.
    if COMM.rank == 0:
        jobs = list(range(100))
        # Split into however many cores are available.
        jobs = split_container(jobs, COMM.size)
    else:
        jobs = None
        
    # Scatter jobs across cores.
    jobs = COMM.scatter(jobs, root=0)
    print(jobs)
