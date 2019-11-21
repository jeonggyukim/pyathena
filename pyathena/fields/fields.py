import astropy.constants as ac
import astropy.units as au

class DerivedFieldsDefault(object):

    func = dict()
    field_dep = dict()
    label = dict()
    
    # nH [cm^-3]
    f = 'nH'
    @staticmethod
    def calc_nH(d, u):
        return d['density']
    func[f] = calc_nH
    field_dep[f] = ['density']
    label[f] = r'$n_{\rm H}\;{\rm cm^{-3}}$'
    
    # nH2 [cm^-3]
    f = 'nH2'
    @staticmethod
    def calc_nH2(d, u):
        return d['density']*d['xH2']
    func[f] = calc_nH2
    field_dep[f] = set(['density', 'xH2'])
    label[f] = r'$n_{\rm H_2}\;{\rm cm^{-3}}$'
    
    # P/kB [K cm^-3]
    f = 'pok'
    @staticmethod
    def calc_pok(d, u):
        return d['pressure']*(u.energy_density/ac.k_B).cgs.value
    func[f] = calc_pok
    field_dep[f] = set(['pressure'])
    label[f] = r'$P/k_{\rm B}\;{\rm cm^{-3}\,K}$'
    
    def __init__(self):

        # Create a dictionary containing all information about derived fields
        self.dfi = dict()
        self.derived_field_list = self.func
        
        for f in self.derived_field_list:
            self.dfi[f] = dict(func=self.func[f].__func__,
                               field_dep=self.field_dep[f],
                               label=self.label[f])
