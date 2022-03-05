from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
from amplitf.kinematics import *
from amplitf.dalitz_decomposition import *
import amplitf.interface as atfi
import matplotlib.pyplot as plt
from amplitf.constants import spin as sp
from amplitf.constants import angular as ang
from matplotlib.colors import LogNorm
import tensorflow as tf
from amplitf.amplitudes.resonances import *


ma = 2286.46 # lambda_c spin = 0.5 parity = 1
mb = 1864.84 # D^0 bar spin = 0 partiy = -1
mc = 493.677 # K-  spin = 0 parity = -1
md = 5619.60  # lambda_b  spin = 0.5 parity = +1
phsp = DalitzPhaseSpace(ma,mb,mc,md) 

smp = PhaseSpaceSample(phsp,phsp.rectangular_grid_sample(200, 200, space_to_sample="linDP"))


sgma3 = phsp.m2ab(smp) # lmbda_c , D_bar
sgma2 = phsp.m2ac(smp) # lmbda_c , k
sgma1 = phsp.m2bc(smp) # D_bar , k

bls_ds_kmatrix_in = {(0,1):atfi.complex(atfi.const(-1.8),atfi.const(4.4)),
                        (2,1):atfi.complex(atfi.const(-7.05),atfi.const(-4.06)),
                        (2,3):atfi.complex(atfi.const(4.96),atfi.const(-4.73))}
bls_ds_kmatrix_out = {(2,0):atfi.complex(atfi.const(-1.064),atfi.const(-0.722))}
alphas = [atfi.complex(atfi.const(0.00272),atfi.const(-0.00715)), atfi.complex(atfi.const(-0.00111),atfi.const(0.00394)),atfi.complex(atfi.const(1),atfi.const(0))]
g0,g1,g2,g3 = -8.73, 6.54,6.6,-3.38
m11,m12,m21,m22 = mb,mc,1863,mc 
channels = [
    KmatChannel(m11,m12,sp.SPIN_1,0.0135,index=0), # this is the decay channel we will see
    # KmatChannel(m11,m12,sp.SPIN_0,0.0135,index=0), # this is the decay channel we will see
    KmatChannel(m21,m22,sp.SPIN_1,0.0867,index=1), # this is a channel that may cause interference
    # KmatChannel(m21,m22,sp.SPIN_0,0.0867,index=1) # this is a channel that may cause interference
    #KmatChannel(2420,mc,sp.SPIN_1,0.9,index=2), # this is a channel that may cause interference
]
poles = [
    KmatPole(2713.6,[g0,g1,5]),  # D^*_s1(2700)
    KmatPole(2967.1,[g2,g3,5])  # D^*_s1(2860)    # ToDo find if we assigned the g values correctly #D^*_s1(2860)
]
D_kma = kmatrix(sp.SPIN_1,-1,5./1000.,alphas,channels,poles,
                        bls_ds_kmatrix_in,bls_ds_kmatrix_out,out_channel=0)


if __name__ =="__main__":
    s1_plt = atfi.convert_to_tensor(np.linspace(min(sgma1),max(sgma1),10000))
    ampl = abs(D_kma.X(s1_plt,2))**2
    plt.plot(s1_plt**0.5/1e3,ampl)
    plt.show()