import numpy as np
import pandas as pd
import os

class TimingReader(object):
    def __init__(self, basedir, problem_id):
        """ Timing reader class

        Parameters
        ----------
        basedir : str
            base directory name
        problem_id : str
            problem id

        """
        self.fdict = dict()
        lt = os.path.join(basedir, '{}.loop_time.txt'.format(problem_id))
        tt = os.path.join(basedir, '{}.task_time.txt'.format(problem_id))
        if os.path.isfile(lt):
            self.fdict['loop_time'] = lt
        if os.path.isfile(tt):
            self.fdict['task_time'] = tt

    def load_task_time(self, groups=None):
        """Read .task_time.txt file

        Parameters
        ----------
        groups : list
            If provided, group tasks that have the same string in the list
            everything else will be summed and stored in 'Others'.
            e.g., ['Hydro','Primitives','UserWork']

        Returns
        -------
        pandas.DataFrame
            The breakdown of time taken by each task of the time integrator
        """
        def from_block(block):
            info = dict()
            h = block[0].split(',')
            info['ncycle'] = int(h[0].split('=')[1])
            name = h[1].split('=')[1]
            time = h[2].split('=')[1]
            info[name] = float(time)
            for line in block[1:]:
                sp = line.split(',')
                name = sp[0].replace(' ', '')
                time = sp[1].split('=')[1]
                info[name] = float(time)
            return info

        with open(self.fdict['task_time']) as fp:
            lines = fp.readlines()

        block_idx = []
        for i, line in enumerate(lines):
            if line.startswith('#'):
                block_idx.append(i)
        timing = dict()

        # initialize
        info = from_block(lines[0:block_idx[1]])

        if groups is None:
            for k in info:
                timing[k] = []
        else:
            meta = set(['TimeIntegrator', 'ncycle'])
            keys = set(info.keys()) - meta
            members = dict()
            for g in groups:
                members[g] = []
                for i, k in enumerate(info):
                    if g in k:
                        members[g].append(k)
                        keys = keys - set([k])
            members['Others'] = keys
            for k in list(meta) + list(members.keys()):
                timing[k] = []

        for i, j in zip(block_idx[:-1], block_idx[1:]):
            info = from_block(lines[i:j])
            if groups is None:
                for k, v in info.items():
                    timing[k].append(v)
            else:
                for g in members:
                    gtime = 0
                    for k in members[g]:
                        gtime += info[k]
                    timing[g].append(gtime)
                for k in meta:
                    timing[k].append(info[k])

        for k in timing:
            timing[k] = np.array(timing[k])
        return pd.DataFrame(timing)

    def load_loop_time(self):
        """Read .loop_time.txt file

        Returns
        -------
        pandas.DataFrame
            The breakdown of each step of the main loop including
            Before, TimeIntegratorTaskList, SelfGravity, After
        """
        def from_one_line(line):
            info = dict()
            for sp in line.split(','):
                name, value = sp.split('=')
                if name in ['ncycle', 'Nblocks']:
                    info[name] = int(value)
                else:
                    info[name] = float(value)
            return info
        with open(self.fdict['loop_time']) as fp:
            lines = fp.readlines()

        timing = dict()
        info = from_one_line(lines[0])
        for k in info:
            timing[k] = []

        for line in lines:
            info = from_one_line(line)
            for k, v in info.items():
                timing[k].append(v)
        return pd.DataFrame(timing).set_index('ncycle')
