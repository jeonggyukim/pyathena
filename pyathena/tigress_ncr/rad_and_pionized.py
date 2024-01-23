# rad_and_pionized.py

import os.path as osp
import numpy as np

from ..load_sim import LoadSim

class RadiationAndPartiallyIonized:
    """
    Vertical profile of Lyman continuum radiation field and partially ionized gas
    """
    @LoadSim.Decorators.check_pickle
    def read_zprof_Erad_LyC(self, num, prefix='Erad_LyC',
                            savdir=None, force_override=False):
        ds = self.load_vtk(num)
        dd = ds.get_field(['nH','T','xHII','xe','ne','nesq','Erad_LyC','Uion'])
        NxNy = ds.domain['Nx'][0]*ds.domain['Nx'][1]
        area_tot = ds.domain['Lx'][0]*ds.domain['Lx'][1]
        dx = ds.domain['dx'][0]

        idx = dict()
        cond = dict()
        f_cond = lambda dd, f, op, v: op(dd[f], v)

        cond['wT0'] = ['T', np.greater_equal, 3.0e3]
        cond['wT1'] = ['T', np.less, 6.0e3]
        cond['wT2'] = ['T', np.less, 1.5e4]
        cond['wT3'] = ['T', np.less, 3.5e4]
        cond['ion'] = ['xHII', np.greater, 0.9]
        cond['neu'] = ['xHII', np.less_equal, 0.1]
        cond['LyC'] = ['Erad_LyC', np.greater, 0.0]
        cond['Uion_pi'] = ['Uion', np.greater, 1e-5]
        cond['xHII_ge_0.5'] = ['xHII', np.greater_equal, 0.5]

        # Indices
        for k,v in cond.items():
            idx[k] = f_cond(dd, *v)

        idx['pion'] = np.logical_and(~idx['ion'], ~idx['neu'])
        idx['w'] = np.logical_and(~idx['wT1'], idx['wT3'])
        idx['wion'] = np.logical_and(idx['w'], idx['xHII_ge_0.5'])
        idx['wion_LyC'] = np.logical_and(idx['wion'], idx['LyC'])

        # Select cells and data
        # non-zero Erad_LyC
        Erad_LyC_LyC = np.ma.masked_invalid(dd.where(idx['LyC'])['Erad_LyC'].data)
        Uion_LyC = np.ma.masked_invalid(dd.where(idx['LyC'])['Uion'].data)
        # non-zero Erad_LyC and Uion > Uion_pi
        Erad_LyC_LyC_Uion_pi = np.ma.masked_invalid(dd.where(
            np.logical_and(idx['LyC'], idx['Uion_pi']))['Erad_LyC'].data)

        # Partially ionized
        nesq_pion = np.ma.masked_invalid(dd.where(idx['pion'])['nesq'].data)
        ne_pion = np.ma.masked_invalid(dd.where(idx['pion'])['ne'].data)
        Uion_pion = np.ma.masked_invalid(dd.where(idx['pion'])['Uion'].data)

        # Warm ionized (xHII > 0.5)
        nesq_wion = np.ma.masked_invalid(dd.where(idx['wion'])['nesq'].data)
        ne_wion = np.ma.masked_invalid(dd.where(idx['wion'])['ne'].data)
        Uion_wion = np.ma.masked_invalid(dd.where(idx['wion'])['Uion'].data)

        # Warm ionized and exposed to LyC
        nesq_wion_LyC = np.ma.masked_invalid(dd.where(idx['wion_LyC'])['nesq'].data)
        ne_wion_LyC = np.ma.masked_invalid(dd.where(idx['wion_LyC'])['ne'].data)
        Uion_wion_LyC = np.ma.masked_invalid(dd.where(idx['wion_LyC'])['Uion'].data)

        r = dict(
            # Filling factor
            f_area_LyC=Erad_LyC_LyC.count(axis=(1,2)) / NxNy,
            f_area_LyC_Uion_pi=Erad_LyC_LyC_Uion_pi.count(axis=(1,2)) / NxNy,
            f_area_pion=ne_pion.count(axis=(1,2)) / NxNy,
            f_area_wion=ne_wion.count(axis=(1,2)) / NxNy,
            f_area_wion_LyC=ne_wion_LyC.count(axis=(1,2)) / NxNy,
            # Area-averaged quantities
            nesq_wion_LyC=nesq_wion_LyC.sum(axis=(1,2)) / NxNy,
            nesq_wion=nesq_wion.sum(axis=(1,2)) / NxNy,
            nesq_pion=nesq_pion.sum(axis=(1,2)) / NxNy,
            ne_pion=ne_pion.sum(axis=(1,2)) / NxNy,
            Erad_LyC_LyC=Erad_LyC_LyC.sum(axis=(1,2)) / NxNy,
            # All gas
            Erad_LyC=dd['Erad_LyC'].sum(axis=(1,2)) / NxNy,
            nesq=dd['nesq'].sum(axis=(1,2)) / NxNy,
            ne=dd['ne'].sum(axis=(1,2)) / NxNy,
            Uion=dd['Uion'].sum(axis=(1,2)) / NxNy,
            # nesq-weighted Uion
            Uion_wgt_nesq=(dd['Uion']*dd['nesq']).sum(axis=(1,2))/dd['nesq'].sum(axis=(1,2)),
            # nesq-weighted Uion for wion_LyC
            Uion_wion_LyC_wgt_nesq=(dd['Uion_wion_LyC']*dd['nesq_wion_LyC']).sum(axis=(1,2))/\
            dd['nesq_wion_LyC'].sum(axis=(1,2)),
            # ne-weighted Uion for wion_LyC
            Uion_wion_LyC_wgt_ne=(dd['Uion_wion_LyC']*dd['ne_wion_LyC']).sum(axis=(1,2))/\
            dd['ne_wion_LyC'].sum(axis=(1,2)),
            # vol-weighted Uion for wion_LyC
            Uion_wion_LyC=(dd['Uion_wion_LyC']*dd['ne_wion_LyC']).sum(axis=(1,2))/\
            dd['ne_wion_LyC'].sum(axis=(1,2)),
            # ne-weighted Uion
            Uion_wgt_ne=(dd['Uion']*dd['ne']).sum(axis=(1,2))/dd['ne'].sum(axis=(1,2)),
            # nesq-weighted xHII
            xHII_wgt_nesq=(dd['xHII']*dd['nesq']).sum(axis=(1,2))/dd['nesq'].sum(axis=(1,2)),
            # ne-weighted xHII
            xHII_wgt_ne=(dd['xHII']*dd['ne']).sum(axis=(1,2))/dd['ne'].sum(axis=(1,2)),
            z=dd.z.values, NxNy=NxNy,
            time=ds.domain['time']
        )

        return r

    def get_zprof_Erad_LyC_all(self, nums,
                               savdir=None, force_override=False):
        if savdir is None:
            savdir = osp.join(self.savdir, 'Erad_LyC')

        print(savdir)
        rr = dict()
        for i,num in enumerate(nums):
            print(num, end=' ')
            r = self.read_zprof_Erad_LyC(num, savdir=savdir, force_override=False)
            if i == 0:
                for k in r.keys():
                    if k == 'time':
                        rr[k] = []
                    else:
                        rr[k] = []

            for k in r.keys():
                if k == 'time':
                    rr[k].append(r['time'])
                elif k == 'nbins' or k=='NxNy' or k == 'bins' or k == 'domain':
                    rr[k] = r[k]
                else:
                    rr[k].append(r[k])

        for k in rr.keys():
            if k == 'time' or k == 'nbins' or k == 'NxNy' or k == 'bins' or k == 'domain':
                continue
            else:
                rr[k] = np.stack(rr[k], axis=0)

        rr['time'] = np.array(rr['time'])

        return rr

    @LoadSim.Decorators.check_pickle
    def read_zprof_partially_ionized(self, num, prefix='pionized',
                                     savdir=None, force_override=False):
        """
        Compute z-profile of gas binned by xHII
        """
        ds = self.load_vtk(num)
        dd = ds.get_field(['nH','T','xHII','ne','nesq','Erad_LyC','Uion'])

        NxNy = ds.domain['Nx'][0]*ds.domain['Nx'][1]

        # Bin by xHII
        # nbins = 11
        bins = np.linspace(0, 1, num=nbins)
        #bins = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
        #bins = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        nbins = len(bins)

        idx = []
        area = []
        ne_ma = []
        nesq_ma = []
        Erad_LyC_ma = []
        Uion_ma = []
        ne = []
        nesq = []
        Erad_LyC = []
        Uion = []

        idx_w1 = []
        area_w1 = []
        ne_ma_w1 = []
        nesq_ma_w1 = []
        Erad_LyC_ma_w1 = []
        Uion_ma_w1 = []
        ne_w1 = []
        nesq_w1 = []
        Erad_LyC_w1 = []
        Uion_w1 = []

        idx_w2 = []
        area_w2 = []
        ne_ma_w2 = []
        nesq_ma_w2 = []
        Erad_LyC_ma_w2 = []
        Uion_ma_w2 = []
        ne_w2 = []
        nesq_w2 = []
        Erad_LyC_w2 = []
        Uion_w2 = []

        for i in range(nbins-1):
            if i == 0:
                # Select gas based on xHII only
                idx.append((bins[i] <= dd['xHII']) & (dd['xHII'] <= bins[i+1]))
                # w1: Select gas based on xHII and 3e3 < T < 1.5e4
                idx_w1.append((bins[i] <= dd['xHII']) & (dd['xHII'] <= bins[i+1]) &
                              (dd['T'] >= 3.0e3) & (dd['T'] < 1.5e4))
                # w2: Select gas based on xHII and 1.5e4 < T < 3.5e4
                idx_w2.append((bins[i] <= dd['xHII']) & (dd['xHII'] <= bins[i+1]) &
                              (dd['T'] >= 1.5e4) & (dd['T'] < 3.5e4))
            else:
                idx.append((bins[i] < dd['xHII']) & (dd['xHII'] <= bins[i+1]))
                idx_w1.append((bins[i] < dd['xHII']) & (dd['xHII'] <= bins[i+1]) &
                              (dd['T'] >= 3.0e3) & (dd['T'] < 1.5e4))
                idx_w2.append((bins[i] < dd['xHII']) & (dd['xHII'] <= bins[i+1]) &
                              (dd['T'] >= 1.5e4) & (dd['T'] < 3.5e4))

            ne_ma.append(np.ma.masked_invalid(dd.where(idx[i])['ne'].data))
            nesq_ma.append(np.ma.masked_invalid(dd.where(idx[i])['nesq'].data))
            Erad_LyC_ma.append(np.ma.masked_invalid(dd.where(idx[i])['Erad_LyC'].data))
            Uion_ma.append(np.ma.masked_invalid(dd.where(idx[i])['Uion'].data))

            ne_ma_w1.append(np.ma.masked_invalid(dd.where(idx_w1[i])['ne'].data))
            nesq_ma_w1.append(np.ma.masked_invalid(dd.where(idx_w1[i])['nesq'].data))
            Erad_LyC_ma_w1.append(np.ma.masked_invalid(dd.where(idx_w1[i])['Erad_LyC'].data))
            Uion_ma_w1.append(np.ma.masked_invalid(dd.where(idx_w1[i])['Uion'].data))

            ne_ma_w2.append(np.ma.masked_invalid(dd.where(idx_w2[i])['ne'].data))
            nesq_ma_w2.append(np.ma.masked_invalid(dd.where(idx_w2[i])['nesq'].data))
            Erad_LyC_ma_w2.append(np.ma.masked_invalid(dd.where(idx_w2[i])['Erad_LyC'].data))
            Uion_ma_w2.append(np.ma.masked_invalid(dd.where(idx_w2[i])['Uion'].data))

            area_tot = ds.domain['Lx'][0]*ds.domain['Lx'][1]
            dx = ds.domain['dx'][0]

            area.append(ne_ma[i].count(axis=(1,2))*dx**2)
            ne.append(ne_ma[i].sum(axis=(1,2)))
            nesq.append(nesq_ma[i].sum(axis=(1,2)))
            Erad_LyC.append(Erad_LyC_ma[i].sum(axis=(1,2)))
            Uion.append(Uion_ma[i].sum(axis=(1,2)))

            area_w1.append(ne_ma_w1[i].count(axis=(1,2))*dx**2)
            ne_w1.append(ne_ma_w1[i].sum(axis=(1,2)))
            nesq_w1.append(nesq_ma_w1[i].sum(axis=(1,2)))
            Erad_LyC_w1.append(Erad_LyC_ma_w1[i].sum(axis=(1,2)))
            Uion_w1.append(Uion_ma_w1[i].sum(axis=(1,2)))

            area_w2.append(ne_ma_w2[i].count(axis=(1,2))*dx**2)
            ne_w2.append(ne_ma_w2[i].sum(axis=(1,2)))
            nesq_w2.append(nesq_ma_w2[i].sum(axis=(1,2)))
            Erad_LyC_w2.append(Erad_LyC_ma_w2[i].sum(axis=(1,2)))
            Uion_w2.append(Uion_ma_w2[i].sum(axis=(1,2)))

        # Area-averaged quantities <.>
        area = np.array(area)
        f_area = area / area.sum(axis=0)
        ne = np.array(ne) / NxNy
        nesq = np.array(nesq) / NxNy
        Erad_LyC = np.array(Erad_LyC) / NxNy
        Uion = np.array(Uion) / NxNy

        area_w1 = np.array(area_w1)
        f_area_w1 = area_w1 / area.sum(axis=0)
        ne_w1 = np.array(ne_w1) / NxNy
        nesq_w1 = np.array(nesq_w1) / NxNy
        Erad_LyC_w1 = np.array(Erad_LyC_w1) / NxNy
        Uion_w1 = np.array(Uion_w1) / NxNy

        area_w2 = np.array(area_w2)
        f_area_w2 = area_w2 / area.sum(axis=0)
        ne_w2 = np.array(ne_w1) / NxNy
        nesq_w2 = np.array(nesq_w2) / NxNy
        Erad_LyC_w2 = np.array(Erad_LyC_w2) / NxNy
        Uion_w2 = np.array(Uion_w2) / NxNy

        r = dict(area=area, area_w1=area_w1, area_w2=area_w2,
                 f_area=f_area, f_area_w1=f_area_w1, f_area_w2=f_area_w2,
                 ne=ne, ne_w1=ne_w1, ne_w2=ne_w2,
                 nesq=nesq, nesq_w1=nesq_w1, nesq_w2=nesq_w2,
                 Erad_LyC=Erad_LyC, Erad_LyC_w1=Erad_LyC_w1, Erad_LyC_w2=Erad_LyC_w2,
                 Uion=Uion, Uion_w1=Uion_w1, Uion_w2=Uion_w2,
                 bins=bins, nbins=nbins, NxNy=NxNy,
                 time=ds.domain['time'],
                 domain=ds.domain)

        return r

    def get_zprof_partially_ionized_all(self, nums,
                                        savdir=None, force_override=False):
        if savdir is None:
            savdir = osp.join(self.savdir, 'pionized')

        rr = dict()
        for i,num in enumerate(nums):
            print(num, end=' ')
            r = self.read_zprof_partially_ionized(num,
                        savdir=savdir, force_override=False)
            if i == 0:
                for k in r.keys():
                    if k == 'time':
                        rr[k] = []
                    else:
                        rr[k] = []

            for k in r.keys():
                if k == 'time':
                    rr[k].append(r['time'])
                elif k == 'nbins' or k=='NxNy' or k == 'bins' or k == 'domain':
                    rr[k] = r[k]
                else:
                    rr[k].append(r[k])

        for k in rr.keys():
            if k == 'time' or k == 'nbins' or k == 'NxNy' or k == 'bins' or k == 'domain':
                continue
            else:
                rr[k] = np.stack(rr[k], axis=0)

        rr['time'] = np.array(rr['time'])

        return rr
