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

    @staticmethod
    def _read_block(block):
        res = dict()
        h = block[0].split(',')
        res['ncycle'] = int(h[0].split('=')[1])
        name = h[1].split('=')[1]
        name_tlist = name
        time = h[2].split('=')[1]
        res[name] = float(time)
        for line in block[1:]:
            sp = line.split(',')
            name = sp[0].replace(' ', '')
            time = sp[1].split('=')[1]
            res[name] = float(time)
        return res, name_tlist

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
        dictionary of pandas.DataFrame
            The breakdown of time taken by each task of the time integrator
        """

        with open(self.fdict['task_time']) as fp:
            lines = fp.readlines()

        block_idx = []
        for i, line in enumerate(lines):
            if line.startswith('#'):
                block_idx.append(i)
        tt = {}
        names = []
        for i in range(len(block_idx)-1):
            block, name = self._read_block(lines[block_idx[i]:block_idx[i+1]])
            if not name in names:
                names.append(name)
                tt[name] = {}
                for f in block.keys():
                    tt[name][f] = []

            for k, v in block.items():
                tt[name][k].append(v)

        dfa = {}
        for name in names:
            df = pd.DataFrame(tt[name])
            df['All'] = df.loc[:, ~df.columns.str.contains('ncycle')].sum(axis=1)
            if groups is not None:
                df['Others'] = df.loc[:, ~df.columns.str.contains(
                    '|'.join(groups + ['ncycle', 'All']))].sum(axis=1)
                for g in groups:
                    df[g] = df.filter(like=g).sum(axis=1)
            dfa[name] = df

        return dfa

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
