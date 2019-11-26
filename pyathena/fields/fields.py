import astropy.constants as ac
import astropy.units as au

class DerivedFieldsDefault(object):

    func = dict()
    field_dep = dict()
    label = dict()
    
    # nH [cm^-3] (assume d=nH)
    f = 'nH'
    field_dep[f] = ['density']
    @staticmethod
    def calc_nH(d, u):
        return d['density']
    func[f] = calc_nH
    label[f] = r'$n_{\rm H}\;{\rm cm^{-3}}$'

    # rho [g cm^-3]
    f = 'rho'
    field_dep[f] = ['density']
    @staticmethod
    def calc_rho(d, u):
        return d['density']*(u.muH*u.mH).cgs.value
    func[f] = calc_rho
    label[f] = r'$\rho\;{\rm g\,cm^{-3}}$'

    # nH2 [cm^-3] (assume d=nH)
    f = 'nH2'
    field_dep[f] = set(['density', 'xH2'])
    @staticmethod
    def calc_nH2(d, u):
        return d['density']*d['xH2']
    func[f] = calc_nH2
    label[f] = r'$n_{\rm H_2}\;{\rm cm^{-3}}$'
    
    # P/kB [K cm^-3]
    f = 'pok'
    field_dep[f] = set(['pressure'])
    @staticmethod
    def calc_pok(d, u):
        return d['pressure']*(u.energy_density/ac.k_B).cgs.value
    func[f] = calc_pok
    label[f] = r'$P/k_{\rm B}\;{\rm cm^{-3}\,K}$'
    
    def __init__(self):

        # Create a dictionary containing all information about derived fields
        self.dfi = dict()
        self.derived_field_list = self.func
        
        for f in self.derived_field_list:
            self.dfi[f] = dict(field_dep=self.field_dep[f],
                               func=self.func[f].__func__,
                               label=self.label[f])
