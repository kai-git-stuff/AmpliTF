from amplitf.likelihood import *
import amplitf.interface as atfi
from numpy import isclose

def test_integrals():
    pdf = atfi.range(0,1000,dtype=atfi.fptype())

    assert integral(pdf) == 499.5

    weights = atfi.linspace(1,0,1000)

    assert abs(weighted_integral(pdf,weights) - 166.33333333333) < 1e-5




    




