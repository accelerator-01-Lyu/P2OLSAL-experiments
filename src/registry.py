"""Method registry: 16 methods (6 P2 variants + 6 SOTA + 4 fuzzy baselines)."""
from .p2_method import P2OLSAL
from .baselines_sota import (RandomSampler, EntropySampler, BADGESampler,
                             CoreSetSampler, BALDSampler, QBCSampler,
                             RepresentativePassiveSampler)
from .baselines_fuzzy import SSFCM, CEFCM, GRFCM, PLCFCMPassive
from .plcfcm import plcfcm_fit


def _make_p2(robust=True, exploration=True, redundancy=True, fmis=True,
             block=True, name='P2_Full'):
    def factory():
        return P2OLSAL(robust=robust, exploration=exploration,
                       redundancy=redundancy, fmis=fmis, block=block,
                       name=name)
    return factory


def _make_sota(cls, name=None):
    def factory():
        obj = cls()
        if name:
            obj.name = name
        return obj
    return factory


def _make_fuzzy(solver_cls):
    def factory():
        return solver_cls()
    return factory


def _state_plcfcm(dataset, c, seed):
    """State: base clusterer is PLCFCM, sampler is the method's select()."""
    return {'base_solver': 'plcfcm', 'c': c, 'seed': seed}


def _state_fuzzy(solver_name):
    def statef(dataset, c, seed):
        return {'base_solver': solver_name, 'c': c, 'seed': seed}
    return statef


def build_registry():
    reg = {}
    # P2 variants (6)
    reg['P2_Full'] = (_make_p2(name='P2_Full'), _state_plcfcm)
    reg['P2_NoRobust'] = (_make_p2(robust=False, name='P2_NoRobust'), _state_plcfcm)
    reg['P2_NoExploration'] = (_make_p2(exploration=False, name='P2_NoExploration'), _state_plcfcm)
    reg['P2_NoRedundancy'] = (_make_p2(redundancy=False, name='P2_NoRedundancy'), _state_plcfcm)
    reg['P2_NoFMIS'] = (_make_p2(fmis=False, name='P2_NoFMIS'), _state_plcfcm)
    reg['P2_NoBlock'] = (_make_p2(block=False, name='P2_NoBlock'), _state_plcfcm)
    # SOTA baselines (6) - all use PLCFCM as base solver
    reg['Random'] = (_make_sota(RandomSampler, 'Random'), _state_plcfcm)
    reg['Entropy'] = (_make_sota(EntropySampler, 'Entropy'), _state_plcfcm)
    reg['BADGE'] = (_make_sota(BADGESampler, 'BADGE'), _state_plcfcm)
    reg['CoreSet'] = (_make_sota(CoreSetSampler, 'CoreSet'), _state_plcfcm)
    reg['BALD'] = (_make_sota(BALDSampler, 'BALD'), _state_plcfcm)
    reg['QBC'] = (_make_sota(QBCSampler, 'QBC'), _state_plcfcm)
    # Fuzzy baselines (4) - use entropy sampling with alternative solvers
    reg['SSFCM'] = (_make_sota(EntropySampler, 'SSFCM'), _state_fuzzy('ssfcm'))
    reg['CEFCM'] = (_make_sota(EntropySampler, 'CEFCM'), _state_fuzzy('ceffcm'))
    reg['GRFCM'] = (_make_sota(EntropySampler, 'GRFCM'), _state_fuzzy('grfcm'))
    reg['PLCFCMPassive'] = (_make_sota(RepresentativePassiveSampler, 'PLCFCMPassive'), _state_fuzzy('plcfcm_passive'))
    return reg
