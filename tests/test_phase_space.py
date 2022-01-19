import numpy as np
import amplitf.interface as atfi


def test_phase_space_sample():
    from amplitf.phasespace.base_phasespace import PhaseSpaceSample
    from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace

    phsp = DalitzPhaseSpace(149,149,998,2400)

    sample = PhaseSpaceSample(phsp)

    smp = phsp.unfiltered_sample(1000)

    sample.setSample(smp)

    sample = sample.filter()

    filtered_twice_sample = sample.filter()

    assert(len(sample.get_m2ab()) == len(filtered_twice_sample.get_m2bc()))

    assert all(abs(phsp.cos_helicity_bc(sample)) <= 1.0)

if __name__ == "__main__":
    test_phase_space_sample()
