import numpy as np
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Phase:
    """Phase info. Should be mutually exclusive.
    """
    name: str
    # Used to mask data
    id: int
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

    def __post_init__(self):
        object.__setattr__(self, 'num_phases', len(self.phases))
        object.__setattr__(self, 'phase_names', [ph.name for ph in self.phases])

def create_phase_set_with_LyC_mask(ph_set):
    import copy
    phases = []
    for ph in ph_set.phases:
        cond0 = copy.deepcopy(ph.cond)
        cond0.append(['Erad_LyC_mask', np.equal, 0.0])
        phases.append(Phase(ph.name + '_noLyC', ph.id, cond0))

    for ph in ph_set.phases:
        cond1 = copy.deepcopy(ph.cond)
        cond1.append(['Erad_LyC_mask', np.equal, 1.0])
        phases.append(Phase(ph.name + '_LyC', ph.id + ph_set.num_phases, cond1))

    return PhaseSet(ph_set.name + '_LyC_ma', phases)
