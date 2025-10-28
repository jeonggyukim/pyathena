import glob
import os
import re
import os.path as osp
import numpy as np

from .logger import create_logger

from .io.read_athinput import read_athinput
from .io.athena_read import athinput


class FindFiles(object):

    # Default patterns
    # TODO: more explicit glob patterns using problem_id as in rst?
    patterns = dict()

    patterns['athinput'] = [
        ('athinput.runtime',),
        ('out.txt',),
        ('out*.txt',),
        ('stdout.txt',),
        ('log.txt',),
        ('*.par',),
        ('*.out',),
        ('athinput.*',),
        ('slurm-*',)]

    patterns['hst'] = [
        ('id0', '*.hst'),
        ('hst', '*.hst'),
        ('*.hst',)]

    patterns['sphst'] = [
        ('id0', '*.star'),
        ('hst', '*.star'),
        ('*.star',)]

    patterns['sn'] = [
        ('id0', '*.sn'),
        ('hst', '*.sn'),
        ('*.sn',)]

    patterns['vtk'] = [
        ('vtk', '*.????.vtk'),
        ('*.????.vtk',)]

    patterns['vtk_id0'] = [
        ('vtk', 'id0', '*.' + '[0-9]'*4 + '.vtk'),
        ('id0', '*.' + '[0-9]'*4 + '.vtk')]

    patterns['vtk_tar'] = [('vtk', '*.????.tar')]

    patterns['vtk_athenapp'] = [
        ('*.block*.out*.?????.vtk',),
        ('*.joined.out*.?????.vtk',)]

    patterns['hdf5'] = [
        ('hdf5', '*.out?.?????.athdf'),
        ('*.out?.?????.athdf',)]

    patterns['starpar_vtk'] = [
        ('starpar', '*.????.starpar.vtk'),
        ('id0', '*.????.starpar.vtk'),
        ('*.????.starpar.vtk',)]

    patterns['partab'] = [
        ('partab', '*.out?.?????.par?.tab'),
        ('*.out?.?????.par?.tab',)]

    patterns['parbin'] = [
        ('parbin', '*.out?.?????.par?.parbin'),
        ('*.out?.?????.par?.parbin',)]

    patterns['parhst'] = [
        ('parhst', '*.par*.csv'),
        ('*.par*.csv',)]

    patterns['zprof'] = [
        ('*.zprof',),
        ('zprof', '*.zprof'),
        ('id0', '*.zprof')]

    patterns['timeit'] = [
        ('timeit.txt',),
        ('timeit', 'timeit.txt')]

    patterns['looptime'] = [('*.loop_time.txt',),]

    patterns['tasktime'] = [('*.task_time.txt',),]

    # patterns for restart defined in find_rst()

    def __init__(self, basedir, verbose=False):
        if not osp.isdir(basedir):
            raise FileNotFoundError('Basedir {0:s} does not exist.'.format(basedir))

        self.logger = create_logger(self.__class__.__name__.split('.')[-1],
                                    verbose)
        self.basedir = basedir
        self.patterns = FindFiles.patterns

        # Find all files
        self.files = dict()
        self.get_basic_info()

        if self.athena_pp:
            self.find_hdf5()

        self.find_vtk()
        if not self.athena_pp:
            self.find_vtk2d()

        self.find_hst()
        self.find_sn()
        self.find_zprof()

        if not self.athena_pp:
            self.find_sphst()
            self.find_starpar_vtk()
        else:
            self.find_partab()
            self.find_parbin()
            self.find_parhst()

        self.find_rst()

        if not self.athena_pp:
            self.find_timeit()
        else:
            self.find_looptime_tasktime()

        # Remove empty file lists
        kdel = [k for k, v in self.files.items() if v == []]
        for k in kdel:
            del self.files[k]

        # Remove empty nums (backward compatibility risk?)
        for attr, v in list(vars(self).items()):
            if attr.startswith('nums') and v == []:
                delattr(self, attr)

    def find_match(self, patterns):
        glob_match = lambda p: sorted(glob.glob(osp.join(self.basedir, *p)))
        for p in patterns:
            f = glob_match(p)
            if f:
                break

        return f

    def get_basic_info(self):
        fathinput = self.find_match(self.patterns['athinput'])
        if fathinput:
            self.files['athinput'] = fathinput[0]
            self.logger.info(f'athinput: {fathinput[0]}')
            try:
                self.par = read_athinput(self.files['athinput'])
            except Exception as e:
                self.logger.warning(f'An error occured: {e}')
                self.logger.warning('Failed to read parameters with read_athinput.' +\
                                    'Try again with athena_read.athinput')
                self.par = athena_read.athinput(self.files['athinput'])
                # TODO: deal with another failure?

            # Determine if it is Athena++ or Athena
            # TODO: determine athena_pp even when par is unavailable
            if 'mesh' in self.par:
                self.athena_pp = True
                self.logger.info('athena_pp: True')
            else:
                self.athena_pp = False
                self.logger.info('athena_pp: False')

            self.out_fmt = []
            self.partags = []
            if self.athena_pp:
                # read output blocks
                for k in self.par.keys():
                    if k.startswith('output'):
                        self.out_fmt.append(self.par[k]['file_type'])

                    # Save particle output tags
                    if k.startswith('particle') and self.par[k]['type'] != 'none':
                        par_id = int(k.strip('particle')) - 1
                        partag = 'par{}'.format(par_id)
                        self.partags.append(partag)

                # if there are hdf5 outputs, save some info
                if self.out_fmt.count('hdf5') > 0:
                    self.hdf5_outid = []
                    self.hdf5_outvar = []
                    for k in self.par.keys():
                        if k.startswith('output') and self.par[k]['file_type'] == 'hdf5':
                            self.hdf5_outid.append(int(re.split(r'(\d+)',k)[1]))
                            self.hdf5_outvar.append(self.par[k]['variable'])
                    for i,v in zip(self.hdf5_outid,self.hdf5_outvar):
                        if 'prim' in v or 'cons' in v:
                            self._hdf5_outid_def = i
                            self._hdf5_outvar_def = v

                # if there are partab outputs, save some info
                if 'partab' in self.out_fmt:
                    for k in self.par.keys():
                        if k.startswith('output') and self.par[k]['file_type'] == 'partab':
                            self.partab_outid = int(re.split(r'(\d+)',k)[1])

                # if there are parbin outputs, save some info
                if 'parbin' in self.out_fmt:
                    for k in self.par.keys():
                        if k.startswith('output') and self.par[k]['file_type'] == 'parbin':
                            self.parbin_outid = int(re.split(r'(\d+)',k)[1])

            else:
                for k in self.par.keys():
                    if k.startswith('output'):
                        # Skip if the block number XX (<outputXX>) is greater than maxout
                        if int(k.replace('output','')) > self.par['job']['maxout']:
                            continue
                        if self.par[k]['out_fmt'] == 'vtk' and \
                           not (self.par[k]['out'] == 'prim' or \
                                self.par[k]['out'] == 'cons'):
                            self.out_fmt.append(self.par[k]['id'] + '.' + \
                                                self.par[k]['out_fmt'])
                        else:
                            self.out_fmt.append(self.par[k]['out_fmt'])

            self.problem_id = self.par['job']['problem_id']
            self.logger.info('problem_id: {0:s}'.format(self.problem_id))
        else:
            # athinput unavailabe
            self.par = None
            self.logger.warning('athinput not found in {0:s}'.\
                                format(self.basedir))
            # TODO: Manually find out_fmt based on existing file extension.
            self.out_fmt = ['hst', 'vtk']

    def find_hst(self):
        if 'hst' in self.out_fmt:
            # Find history dump and extract problem_id (prefix for output file names)
            # Caution: Assumes that problem_id does not contain '.'
            fhst = self.find_match(self.patterns['hst'])
            if fhst:
                self.files['hst'] = fhst[0]
                if not hasattr(self, 'problem_id'):
                    self.problem_id = osp.basename(self.files['hst']).split('.')[:-1]

                self.logger.info('hst: {0:s}'.format(self.files['hst']))
            else:
                self.logger.warning('hst file not found in {0:s}'.\
                                    format(self.basedir))

    def find_sn(self):
        # Find sn dump
        fsn = self.find_match(self.patterns['sn'])
        if fsn:
            self.files['sn'] = fsn[0]
            self.logger.info('sn: {0:s}'.format(self.files['sn']))
        else:
            if not self.athena_pp:
                if self.par is not None:
                    # Issue warning only if iSN is nonzero
                    try:
                        if self.par['feedback']['iSN'] != 0:
                            self.logger.warning('sn file not found in {0:s},' +
                            ' but <feedback>/iSN={1:d}'.\
                            format(self.basedir, self.par['feedback']['iSN']))
                    except KeyError:
                        pass

    def find_sphst(self):
        # Find sphst dump (Athena only)
        fsphst = self.find_match(self.patterns['sphst'])
        if fsphst:
            self.files['sphst'] = fsphst
            self.nums_sphst = [int(f[-10:-5]) for f in self.files['sphst']]
            self.logger.info('sphst: {0:s} nums: {1:d}-{2:d}'.format(
                osp.dirname(self.files['sphst'][0]),
                self.nums_sphst[0], self.nums_sphst[-1]))

    def find_starpar_vtk(self):
        # Find starpar files (Athena only)
        if 'starpar_vtk' in self.out_fmt:
            fstarpar = self.find_match(self.patterns['starpar_vtk'])
            if fstarpar:
                self.files['starpar_vtk'] = fstarpar
                self.nums_starpar = [int(f[-16:-12]) for f in self.files['starpar_vtk']]
                self.logger.info('starpar_vtk: {0:s} nums: {1:d}-{2:d}'.format(
                    osp.dirname(self.files['starpar_vtk'][0]),
                    self.nums_starpar[0], self.nums_starpar[-1]))
            else:
                self.logger.warning(
                    'starpar files not found in {0:s}.'.format(self.basedir))

    def find_partab(self):
        # Find partab files
        if 'partab' in self.out_fmt:
            self.files['partab'] = dict()
            self.nums_partab = dict()
            for partag in self.partags:
                partab_patterns_ = []
                for p in self.patterns['partab']:
                    p = list(p)
                    p[-1] = p[-1].replace('par?', partag)
                    partab_patterns_.append(tuple(p))
                self.files['partab'][partag] = self.find_match(partab_patterns_)
                if not self.files['partab'][partag]:
                    self.logger.warning(
                        'partab ({0:s}) files not found in {1:s}'.\
                        format(partag, self.basedir))
                    self.nums_partab[partag] = None
                else:
                    self.nums_partab[partag] = [int(f[-14:-9])
                                                 for f in self.files['partab'][partag]]
                    self.logger.info('partab ({0:s}): {1:s} nums: {2:d}-{3:d}'.format(
                        partag, osp.dirname(self.files['partab'][partag][0]),
                        self.nums_partab[partag][0], self.nums_partab[partag][-1]))

    def find_parbin(self):
        # Find parbin files
        if 'parbin' in self.out_fmt:
            self.files['parbin'] = dict()
            self.nums_parbin = dict()
            for partag in self.partags:
                parbin_patterns_ = []
                for p in self.patterns['parbin']:
                    p = list(p)
                    p[-1] = p[-1].replace('par?', partag)
                    parbin_patterns_.append(tuple(p))
                self.files['parbin'][partag] = self.find_match(parbin_patterns_)
                if not self.files['parbin'][partag]:
                    self.logger.warning(
                        'parbin ({0:s}) files not found in {1:s}'.\
                        format(partag, self.basedir))
                    self.nums_parbin[partag] = None
                else:
                    self.nums_parbin[partag] = [int(f[-17:-12])
                                                 for f in self.files['parbin'][partag]]
                    self.logger.info('parbin ({0:s}): {1:s} nums: {2:d}-{3:d}'.format(
                        partag, osp.dirname(self.files['parbin'][partag][0]),
                        self.nums_parbin[partag][0], self.nums_parbin[partag][-1]))

    def find_parhst(self):
        if [k for k in self.par.keys() if k.startswith('particle') and
            self.par[k]['type'] != 'none']:
            fparhst = self.find_match(self.patterns['parhst'])
            if fparhst:
                self.files['parhst'] = fparhst
                self.pids = [int(f.split('/')[-1].split('.')[1].strip('par'))
                             for f in self.files['parhst']]
                self.pids.sort()
                self.logger.info('parhst: {0:s} pids: {1:d}-{2:d}'.format(
                    osp.dirname(self.files['parhst'][0]),
                    self.pids[0], self.pids[-1]))
            else:
                self.pids = []
                # TODO: issue warning only when parhst is turned on?
                self.logger.warning(
                    'parhst files not found in {0:s}'.format(self.basedir))

    def find_zprof(self):
        fzprof = self.find_match(self.patterns['zprof'])
        if fzprof:
            self.files['zprof'] = fzprof
            self.nums_zprof = dict()
            self.phase = []
            for f in self.files['zprof']:
                _, num, ph, _ = osp.basename(f).split('.')[-4:]
                if ph in ["pvz","nvz"]:
                    _, num, ph, vz, _ = osp.basename(f).split('.')[-5:]
                    self.zprof_separate_vz = True
                else:
                    self.zprof_separate_vz = False
                try:
                    self.nums_zprof[ph].append(int(num))
                except KeyError:
                    self.phase.append(ph)
                    self.nums_zprof[ph] = []
                    self.nums_zprof[ph].append(int(num))

            # Check if number of files for each phase matches
            num = [len(self.nums_zprof[ph]) for ph in self.nums_zprof.keys()]
            if not all(num):
                self.logger.warning('Number of zprof files doesn\'t match.')
                self.logger.warning(', '.join(['{0:s}: {1:d}'.format(ph, \
                                len(self.nums_zprof[ph])) for ph in self.phase][:-1]))
            else:
                self.logger.info('zprof: {0:s} nums: {1:d}-{2:d}'.format(
                osp.dirname(self.files['zprof'][0]),
                    self.nums_zprof[self.phase[0]][0],
                self.nums_zprof[self.phase[0]][-1]))
        else:
            if 'zprof' in self.out_fmt:
                self.logger.warning(
                    'zprof files not found in {0:s}'.format(self.basedir))

    def find_vtk(self):
        # Find vtk files
        # vtk files in both basedir (joined) and in basedir/id0
        if 'vtk' in self.out_fmt and not self.athena_pp:
            self.files['vtk'] = self.find_match(self.patterns['vtk'])
            self.files['vtk_id0'] = self.find_match(self.patterns['vtk_id0'])
            self.files['vtk_tar'] = self.find_match(self.patterns['vtk_tar'])
            if not self.files['vtk'] and not self.files['vtk_id0'] and \
               not self.files['vtk_tar']:
                self.logger.warning(
                    'vtk files not found in {0:s}'.format(self.basedir))
                self.nums = None
                self.nums_id0 = None
                self.nums_tar = None
                self.nums_vtk = None
            else:
                self.nums = [int(f[-8:-4]) for f in self.files['vtk']]
                self.nums_id0 = [int(f[-8:-4]) for f in self.files['vtk_id0']]
                self.nums_tar = [int(f[-8:-4]) for f in self.files['vtk_tar']]
                if self.nums_id0:
                    self.logger.info('vtk in id0: {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk_id0'][0]),
                        self.nums_id0[0], self.nums_id0[-1]))
                    if not hasattr(self, 'problem_id'):
                        self.problem_id = osp.basename(
                            self.files['vtk_id0'][0]).split('.')[-2:]
                if self.nums:
                    self.logger.info('vtk (joined): {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk'][0]), self.nums[0], self.nums[-1]))
                    if not hasattr(self, 'problem_id'):
                        self.problem_id = osp.basename(
                            self.files['vtk'][0]).split('.')[-2:]
                else:
                    self.nums = self.nums_id0
                if self.nums_tar:
                    self.logger.info('vtk in tar: {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk_tar'][0]),
                        self.nums_tar[0], self.nums_tar[-1]))
                    if not hasattr(self, 'problem_id'):
                        self.problem_id = osp.basename(
                            self.files['vtk_tar'][0]).split('.')[-2:]
                    self.nums = self.nums_tar

                self.nums_vtk = list(set(self.nums) | set(self.nums_id0) |
                                         set(self.nums_tar))
                self.nums_vtk.sort()

            # Check (joined) vtk file size
            sizes = [os.stat(f).st_size for f in self.files['vtk']]
            if len(set(sizes)) > 1:
                size = max(set(sizes), key=sizes.count)
                flist = [(i, s // 1024**2) for i, s in enumerate(sizes) if s != size]
                self.logger.warning('Vtk file size is not unique.')
                for f in flist:
                   self.logger.warning('vtk num: {0:d}, size [MB]: {1:d}'.format(f[0], f[1]))

            # Check (tarred) vtk file size
            sizes = [os.stat(f).st_size for f in self.files['vtk_tar']]
            if len(set(sizes)) > 1:
                size = max(set(sizes), key=sizes.count)
                flist = [(i, s // 1024**2) for i, s in enumerate(sizes) if s != size]
                self.logger.warning('Vtk file size is not unique.')
                for f in flist:
                   self.logger.warning('vtk num: {0:d}, size [MB]: {1:d}'.format(f[0], f[1]))
        elif 'vtk' in self.out_fmt and self.athena_pp:
            # Athena++ vtk files
            self.files['vtk'] = self.find_match(self.patterns['vtk_athenapp'])
            self.nums_vtk = list(set([int(f[-9:-4]) for f in self.files['vtk']]))
            self.nums_vtk.sort()

    def find_hdf5(self):
        # Find hdf5 files
        # hdf5 files in basedir
        if 'hdf5' in self.out_fmt:
            self.files['hdf5'] = dict()
            self.nums_hdf5 = dict()
            for i, v in zip(self.hdf5_outid, self.hdf5_outvar):
                hdf5_patterns_ = []
                for p in self.patterns['hdf5']:
                    p = list(p)
                    p[-1] = p[-1].replace('out?', 'out{0:d}'.format(i))
                    hdf5_patterns_.append(tuple(p))
                self.files['hdf5'][v] = self.find_match(self.patterns['hdf5'])
                if not self.files['hdf5'][v]:
                    self.logger.warning(
                        'hdf5 ({0:s}) files not found in {1:s}'.\
                        format(v, self.basedir))
                    self.nums_hdf5[v] = None
                else:
                    self.nums_hdf5[v] = [int(f[-11:-6]) \
                                         for f in self.files['hdf5'][v]]
                    self.logger.info('hdf5 ({0:s}): {1:s} nums: {2:d}-{3:d}'.format(
                        v, osp.dirname(self.files['hdf5'][v][0]),
                        self.nums_hdf5[v][0], self.nums_hdf5[v][-1]))
                    if not hasattr(self, 'problem_id'):
                        self.problem_id = osp.basename(
                            self.files['hdf5'][self._hdf5_outvar_def][0]).split('.')[-2:]
            # Set nums array
            self.nums = self.nums_hdf5[self._hdf5_outvar_def]

    def find_rst(self):
        # Find rst files
        if 'rst' in self.out_fmt:
            if hasattr(self, 'problem_id'):
                self.patterns['rst'] = [
                    ('rst','{}.*.rst'.format(self.problem_id)),
                    ('rst','????','{}.*.rst'.format(self.problem_id)),
                    ('rst','{}.*.tar'.format(self.problem_id)),
                    ('id0','{}.*.rst'.format(self.problem_id)),
                    ('{}.*.rst'.format(self.problem_id),)]
                frst = self.find_match(self.patterns['rst'])
                if frst:
                    self.files['rst'] = frst
                    if self.athena_pp:
                        numbered_rstfiles = [f for f in self.files['rst'] if 'final' not in f]
                        self.nums_rst = [int(f[-9:-4]) for f in numbered_rstfiles]
                    else:
                        self.nums_rst = [int(f[-8:-4]) for f in self.files['rst']]
                    msg = 'rst: {0:s}'.format(osp.dirname(self.files['rst'][0]))
                    if len(self.nums_rst) > 0:
                        msg += ' nums: {0:d}-{1:d}'.format(
                                self.nums_rst[0], self.nums_rst[-1])
                    if np.any(['final' in f for f in self.files['rst']]):
                        msg += ' final: yes'
                    self.logger.info(msg)
                else:
                    self.logger.warning(
                        'rst files in out_fmt but not found.'.format(self.basedir))

    def find_vtk2d(self):
        # 2d vtk files
        self._fmt_vtk2d_not_found = []
        for fmt in self.out_fmt:
            if '.vtk' in fmt:
                fmt = fmt.split('.')[0]
                vtk2d_patterns = [('id0', '*.????.{0:s}.vtk'.format(fmt)),
                                  ('{0:s}'.format(fmt), '*.????.{0:s}.vtk'.format(fmt))]
                files = self.find_match(vtk2d_patterns)
                if files:
                    self.files[f'{fmt}'] = files
                    setattr(self, f'nums_{fmt}', [int(osp.basename(f).split('.')[1]) \
                                                  for f in self.files[f'{fmt}']])
                else:
                    # Some 2d vtk files may not be found in id0 folder (e.g., slices)
                    self._fmt_vtk2d_not_found.append(fmt)

        if self._fmt_vtk2d_not_found:
            self.logger.info('These vtk files need to be found ' + \
                             'using find_files_vtk2d() method: ' + \
                             ', '.join(self._fmt_vtk2d_not_found))

    def find_timeit(self):
        # Find timeit.txt
        ftimeit = self.find_match(self.patterns['timeit'])
        if ftimeit:
            self.files['timeit'] = ftimeit[0]
            self.logger.info('timeit: {0:s}'.format(self.files['timeit']))
        else:
            self.logger.info('timeit.txt not found.')

    def find_looptime_tasktime(self):
        # Find problem_id.loop_time.txt
        flooptime = self.find_match(self.patterns['looptime'])
        if flooptime:
            self.files['loop_time'] = flooptime[0]
            self.logger.info('loop_time: {0:s}'.format(self.files['loop_time']))
        else:
            self.logger.info('{}.loop_time.txt not found.'.format(self.problem_id))

        # Find problem_id.task_time.txt
        ftasktime = self.find_match(self.patterns['tasktime'])
        if ftasktime:
            self.files['task_time'] = ftasktime[0]
            self.logger.info('task_time: {0:s}'.format(self.files['task_time']))
        else:
            self.logger.info('{}.task_time.txt not found.'.format(self.problem_id))
