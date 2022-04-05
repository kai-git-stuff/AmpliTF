
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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
from amplitf.amplitudes.functional_resonances import *

def phasespace_factor(md,ma,mb):
    # phasespace factor for the dalitz functions
    return 4 * atfi.pi()* atfi.sqrt(two_body_momentum(md,ma,mb)/md)

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
alphas = [atfi.complex(atfi.const(0.00272),atfi.const(-0.00715)), atfi.complex(atfi.const(-0.00111),atfi.const(0.00394))]
g0,g1,g2,g3 = -8.73, 6.54,6.6,-3.38
m11,m12,m21,m22 = mb,mc,2007,mc 
channels = [
    KmatChannel(m11,m12,sp.SPIN_1,0.0135,index=0), # this is the decay channel we will see
    # KmatChannel(m11,m12,sp.SPIN_0,0.0135,index=0), # this is the decay channel we will see
    KmatChannel(m21,m22,sp.SPIN_1,0.0867,index=1), # this is a channel that may cause interference
    # KmatChannel(m21,m22,sp.SPIN_0,0.0867,index=1) # this is a channel that may cause interference
    # KmatChannel(2420,mc,sp.SPIN_1,0.9,index=2), # this is a channel that may cause interference
]
poles = [
    KmatPole(2713.6,[g0,g1]),  # D^*_s1(2700)
    KmatPole(2967.1,[g2,g3])  # D^*_s1(2860)    # ToDo find if we assigned the g values correctly #D^*_s1(2860)
]
D_kma = kmatrix(sp.SPIN_1,-1,1.5/1000.,alphas,channels,poles,
                        bls_ds_kmatrix_in,bls_ds_kmatrix_out,out_channel=0)



m11,m12,m21,m22 = mb,mc,2006.85,mc 
bg1,bg2 = 0.0135, 0.0867
channels = [(0,2,bg1,m11,m12),
            (1,2,bg2,m21,m22)]
resonances = [(2713.6,[g0,g1]),
                (2967.1,[g2,g3])]
# channels = [
#     KmatChannel(m11,m12,2,bg1,index=0), # this is the decay channel we will see
#     KmatChannel(m21,m22,2,bg2,index=1) # this is the channel that may cause interference
# ]
# resonances = [
#     KmatPole(2713.6,[g0,g1]),  # D^*_s1(2700)
#     KmatPole(2967.1,[g2,g3])  # D^*_s1(2860)    # ToDo find if we assigned the g values correctly #D^*_s1(2860)
# ]

def test_Kmat():
    s1_plt = atfi.convert_to_tensor(np.linspace((mb+mc)**2,(md-ma)**2,100000))

    phsp =  phasespace_factor(md,s1_plt**0.5,ma) * phasespace_factor(s1_plt**0.5,mb,mc)

    ampl = abs(D_kma.X(s1_plt))**2
    ampl = ampl
    # plt.plot(s1_plt**0.5/1e3,phasespace_factor(md,s1_plt**0.5,ma),label="1")
    # plt.plot(s1_plt**0.5/1e3,phasespace_factor(s1_plt**0.5,mb,mc),label="2")
    # plt.plot(s1_plt**0.5/1e3,phsp,label="1*2")
    # plt.legend()
    # plt.show()
    print(ampl)
    plt.plot(s1_plt**0.5/1e3,ampl)
    plt.savefig("Kmatrix.png")
    #plt.show()

def test_functional_Kmat():
    s1_plt = atfi.convert_to_tensor(np.linspace((mb+mc)**2,(md-ma)**2,100000))
    m0 = M0(channels,alphas,resonances)
    p0 = two_body_momentum(m0, m11,m12)
    D_kma_func = (sp.SPIN_1,-1,sp.direction_options(sp.SPIN_1),KmatX(channels,alphas,resonances,s1_plt,sp.SPIN_1,0),m0,1.5/1000,p0)
    phsp =  phasespace_factor(md,s1_plt**0.5,ma) * phasespace_factor(s1_plt**0.5,mb,mc)

    ampl = abs(D_kma_func[3])**2
    ampl = ampl
    # plt.plot(s1_plt**0.5/1e3,phasespace_factor(md,s1_plt**0.5,ma),label="1")
    # plt.plot(s1_plt**0.5/1e3,phasespace_factor(s1_plt**0.5,mb,mc),label="2")
    # plt.plot(s1_plt**0.5/1e3,phsp,label="1*2")
    # plt.legend()
    # plt.show()
    print(ampl)
    plt.plot(s1_plt**0.5/1e3,ampl)
    plt.savefig("Kmatrix.png")

if __name__ =="__main__":
    test_functional_Kmat()
    test_Kmat()
    plt.show()
