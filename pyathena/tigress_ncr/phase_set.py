import numpy as np
from dataclasses import dataclass, field
from functools import reduce

@dataclass(frozen=True)
class Phase:
    """Phase info. Need to be mutually exclusive.

    """
    name: str
    # Used to mask data
    mask: int
    # List of conditions for selcting cells by applying reduce(np.logical_and, cond)
    cond: list

@dataclass(frozen=True)
class PhaseSet:
    """List of Phase
    """
    name: str
    phases: list[Phase]
    num_phases: int = field(init=False)
    phase_names: list = field(init=False)
    phase_mask_dict: dict() = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'num_phases', len(self.phases))
        object.__setattr__(self, 'phase_names', [ph.name for ph in self.phases])
        object.__setattr__(self, 'phase_mask_dict',
                           {ph.name: ph.mask for ph in self.phases})

def create_phase_masks(dd, phs):
    # Set phase id and fraction
    v = list(dd.variables)[-1]
    dd = dd.assign(ph_mask=(('z','y','x'), np.zeros_like(dd[v], dtype=int)))

    for ph in phs.phases:
        all_cond = []
        for cond_ in ph.cond:
            all_cond.append((cond_[0])(dd, *(cond_[1:])))

        print(ph.name, ph.mask)
        dd['ph_mask'] += ph.mask*reduce(np.logical_and, all_cond, 1)

    return dd

def create_phase_set_with_density_bins(ph_set,
                                       bins_nH=np.array([-3.0,-2.0,-1.0,0.0,1.0])):
    import copy
    phases = []
    for ph in ph_set.phases:
        cond0 = copy.deepcopy(ph.cond)
        for i in range(len(bins_nH)-1):
            cond0.append(['nH', np.greater_equal, 10.0**bins_nH[i]])
            cond0.append(['nH', np.less, 10.0**bins_nH[i+1]])
            phases.append(Phase(ph.name + '_nH{0:d}'.\
                                format(int(np.log10(10.0**bins_nH[i]))), ph.mask, cond0))

    return PhaseSet(ph_set.name + '_nH', phases)

def create_phase_set_with_LyC_mask(ph_set):
    f1 = lambda dd, v, op, c: op(dd[v], c)
    f2 = lambda dd, v, op, c, v2: op(dd[v], c*dd[v2])
    f3 = lambda dd, v, op, c, v2: op(dd[v], c+dd[v2])

    import copy
    phases = []
    for ph in ph_set.phases:
        cond0 = copy.deepcopy(ph.cond)
        cond0.append([f1, 'Erad_LyC_mask', np.equal, 0.0])
        phases.append(Phase(ph.name + '_noLyC', ph.mask, cond0))

    for ph in ph_set.phases:
        cond1 = copy.deepcopy(ph.cond)
        cond1.append([f1, 'Erad_LyC_mask', np.equal, 1.0])
        cond1.append([f2, 'Uion', np.less, 10.0, 'Uion_pi'])
        phases.append(Phase(ph.name + '_LyC', ph.mask + ph_set.num_phases, cond1))

        # Meaningful for warm photoionized gas
        cond2 = copy.deepcopy(ph.cond)
        cond2.append([f1, 'Erad_LyC_mask', np.equal, 1.0])
        cond2.append([f2, 'Uion', np.greater_equal, 10.0, 'Uion_pi'])
        phases.append(Phase(ph.name + '_LyC_pi', ph.mask + 2*ph_set.num_phases, cond2))

    return PhaseSet(ph_set.name + '_LyC_ma', phases)
