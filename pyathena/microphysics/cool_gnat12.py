"""Read Gnat & Ferland (2012)'s CIE cooling
"""

import os
import os.path as osp
import periodictable as ptbl
import pathlib
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .abundance_solar import AbundanceSolar

class CoolGnat12(object):
    """Class to read ion-by-ion cooling function Gnat & Ferland (2012)

    Properties
    ----------
        info : pandas DataFrame
            basic information
        temp : float array
            tempreature array (10^4 K to 10^8 K)
        cool_cie : float array
            cooling
        ion_frac : float array
            cooling
    """
    def __init__(self, abundance='Asplund09', read_all=True):

        # Directory where Gnat & Ferland (2012) Table 2 and CIE ion fraction
        # table in Gnat & Sternberg (2007) is located
        self.basedir = osp.join(pathlib.Path(__file__).parent.absolute(),
                                '../../data/microphysics')
        try:
            # Read elemental abundances
            self.info = self._read_table2()
            # Read CIE ion fraction
            self.ion_frac = self._read_ion_frac_table()
        except FileNotFoundError:
            print('Download tables with download_data() method')
            raise

        self.temp = self.ion_frac['temp']
        self.cool_cie = {}
        self.cool_cie_per_ion = {}
        if read_all:
            for ion_name in self.info.index:
                self.get_cool_cie(ion_name)

        # Reset abundance
        if abundance == 'Asplund09':
            a = AbundanceSolar(Zprime=1.0)
            for e in self.info.index:
                self.info['abd'][e] = float(a.df[a.df['X'] == e]['NX_NH'])

        # self.cool_cie_tot = self.get_cool_cie_total()
        self.get_cool_cie_total()

    def _read_ion_frac_table(self):
        """Read Gnat & Sternberg (2007) CIE ion fractions
        """

        fname = osp.join(self.basedir, 'Gnat_Sternberg07_cie_ion_frac.txt')
        fp = open(fname,'r')
        lines = fp.readlines()
        fp.close()
        nlines = len(lines[125:])

        fields = ['temp']
        ion_frac = {}
        for l in lines[9:121]:
            ion_name = l.split()[-5].split('{')[0]+ \
                       l[l.rfind('{')+1:
                         max(l.rfind('{')+2,l.rfind('}')-1)].replace('+','1')
            fields.append(ion_name)

        for f in fields:
            ion_frac[f] = np.empty(nlines)

        for iline,l in enumerate(lines[125:]):
            sp = l.split()
            nfields = len(sp)
            if nfields == len(fields):
                for ifield in range(nfields):
                    ion_frac[fields[ifield]][iline] = sp[ifield]

        return pd.DataFrame(ion_frac)

    def _read_table2(self):
        """Read Gnat & Ferland (2012) Table 2
        """

        fname = osp.join(self.basedir, 'Gnat_Ferland12_Table2.txt')
        fp = open(fname,'r')

        lines = fp.readlines()[4:-2]
        fp.close()

        r = {}
        for l in lines:
            sp = l.split('\t')
            elem = ptbl.elements.name(sp[1].lower())
            r[elem.symbol] = {}
            r[elem.symbol]['abd'] = eval(sp[3].replace(' x 10^','e'))
            r[elem.symbol]['name'] = sp[1]
            r[elem.symbol]['number'] = elem.number
            r[elem.symbol]['mass'] = elem.mass
            r[elem.symbol]['datafile'] = osp.join(self.basedir,
                                                  'Gnat_Ferland12_tables', '{}.txt'.\
                                                  format(sp[1]))

        return pd.DataFrame(r).T.sort_values('number')

    def get_cool_cie_total(self,
            elements=['H','He','C','N','O','Ne','Mg','Si','S','Fe']):

        xe = dict()
        xe_tot = np.zeros_like(self.temp)
        cool = dict()
        cool_tot = np.zeros_like(self.temp)

        # Elements for which CIE ion_frac is available

        for e in elements:
            xe[e] = np.zeros_like(self.temp)
            cool[e] = np.zeros_like(self.temp)

        for e in elements:
            nstate = self.info.loc[e]['number'] + 1
            A = self.info.loc[e]['abd']

            for i in range(nstate):
                xe[e] += A*i*self.ion_frac[e + str(i)].values
                cool[e] += A*self.ion_frac[e + str(i)].values*self.cool_cie_per_ion[e][:,i]


        for e in elements:
            xe_tot += xe[e]
            cool_tot += cool[e]

        self.cool_tot = cool_tot
        self.xe_tot = xe_tot
        self.xe = xe
        self.cool = cool

    def get_cool_cie(self, element):
        """
        Parameter
        ---------
        element : str
            Name of ion (e.g., )
        """

        nstate = self.info.loc[element]['number'] + 1
        if not element in self.cool_cie:
            cool_cie_ion, cool_cie = self._read_cool_cie_table(element)
            self.cool_cie[element] = cool_cie
            self.cool_cie_per_ion[element] = cool_cie_ion

        return self.cool_cie[element]

    def _read_cool_cie_table(self, element):
        element = self.info.loc[element]
        nstate = element['number'] + 1
        nskip = nstate + 12
        fp = open(element['datafile'],'r')
        lines = fp.readlines()
        fp.close()
        #print lines[nskip]
        temp = self.temp

        cie = []
        for l in lines[nskip:]:
            cie.append(l.split())
        cie = np.array(cie).astype('float')

        cie_new = np.empty((len(temp),nstate))
        for i in range(nstate):
            cie_new[:,i] = np.interp(temp,cie[:,0],cie[:,i+1])
        cool_tot_new = np.interp(temp,cie[:,0],cie[:,-1])

        return cie_new,cool_tot_new

    def download_data(self, url='http://wise-obs.tau.ac.il/~orlyg/ion_by_ion/'):
        """Download ion-by-ion cooling efficiency tables
        """

        import bs4
        import requests
        datadir = osp.join(self.basedir, 'Gnat_Ferland12_tables')
        if not osp.isdir(datadir):
            os.mkdir(datadir)

        r = requests.get(url)
        data = bs4.BeautifulSoup(r.text, "html.parser")
        for l in data.find_all("a"):
            if l['href'].endswith('txt'):
                r = requests.get(url + l["href"])
                if r.status_code == 200:
                    filename = osp.join(datadir,
                                        '{}.txt'.format(l.contents[0].split()[0]))
                    with open(filename, 'wb') as f:
                        f.write(r.content)
                        print('Downloading... {} to {}'.format(l['href'], filename))

        os.rename(osp.join(datadir,'Aluminium.txt'),
                  osp.join(datadir,'Aluminum.txt'))

    @staticmethod
    def num_to_roman(n):
        """Convert an integer to a Roman numeral
        """

        if not isinstance(n, int):
            raise TypeError("Expected integer, got %s".format(type(n)))
        if not 0 < n < 4000:
            raise ValueError("Argument must be between 1 and 3999")
        ints = (1000,  900,  500,  400, 100,  90, 50,  40, 10,   9,  5,   4, 1)
        nums = ( 'M', 'CM',  'D', 'CD', 'C','XC','L','XL','X','IX','V','IV','I')
        res = []
        for i in range(len(ints)):
            count = int(n/ints[i])
            res.append(nums[i]*count)
            n -= ints[i] * count

        return ''.join(res)

    def plt_ion_frac(self, element, ax=None, **plt_kwargs):
        if ax is None:
            ax = plt.gca()

        for i in range(self.info.loc[element]['number'] +1 ):
            ax.loglog(self.ion_frac['temp'], self.ion_frac['{}{}'.format(element,i)],
                       label='{}{}'.format(element, self.num_to_roman(i+1)), **plt_kwargs)
        # ax.legend()
        plt.setp(ax,xlabel=r'$T [{\rm K}]$')
                 #ylabel=r'{} $X_i$'.format(ptbl.elements.symbol(element).name))

    def plt_cool_cie_ion(self, element, ion_by_ion=True, noplt=False):

        nstate = self.info.loc[element]['number'] + 1
        A = self.info.loc[element]['abd']
        cool_tot = self.get_cool_cie(element)
        if not noplt:
            l, = plt.loglog(self.temp, cool_tot*A,label=element)
            if ion_by_ion and '{}0'.format(element) in self.ion_frac:
                for i in range(nstate):
                    cool_cie_ion = A*self.cool_cie_per_ion[element][:,i]*\
                                  self.ion_frac['{}{}'.format(element, i)]
                    plt.loglog(self.temp, cool_cie_ion, c=l.get_color(), ls=':')

        return cool_tot*A

    def plt_cool_cie_ion(self, element, ion_by_ion=True, noplt=False):

        nstate = self.info.loc[element]['number'] + 1
        A = self.info.loc[element]['abd']
        cool_tot = self.get_cool_cie(element)
        if not noplt:
            l, = plt.loglog(self.temp, cool_tot*A, label=element)
            if ion_by_ion and '{}0'.format(element) in self.ion_frac:
                for i in range(nstate):
                    cool_cie_ion = A*self.cool_cie_per_ion[element][:,i]*\
                                  self.ion_frac['{}{}'.format(element, i)]
                    plt.loglog(self.temp, cool_cie_ion, c=l.get_color(), ls=':')

        return cool_tot*A



    # def plt_cool_cie(self,
    #                  elements=['H','He','C','N','O','Ne','Mg','Si','S','Fe'],
    #                  ax=None, ion_by_ion=True, no_plt=False, **plt_kwargs):
    #     if ax is None:
    #         ax = plt.gca()

    #     cool_tot = np.zeros_like(self.temp)
    #     for element in elements:
    #         cool_tot += plt_cool_cie_ion(tbl, element,
    #                                               ion_by_ion=ion_by_ion, noplt=noplt)
    #     plt.plot(tbl.temp, cool_tot, '-', c='grey', lw=2, **plt_kwargs)
    #     plt.ylim(bottom=1e-25)
    #     if not ion_by_ion:
    #         plt.legend()

    #     return cool_tot
