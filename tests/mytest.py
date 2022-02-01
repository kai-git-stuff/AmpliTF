from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
from amplitf.dynamics import breit_wigner_lineshape
from amplitf.kinematics import *
from amplitf.dalitz_decomposition import *
from amplitf.interface import sqrt
import amplitf.interface as atfi
import matplotlib.pyplot as plt
from amplitf.constants import spin as sp






ma = 2856.1 # lambda_c
mb = 1864.84 # D^0 bar
mc = 493.677 # K-
md = 5912.19 
phsp = DalitzPhaseSpace(ma,mb,mc,md) 

smp = PhaseSpaceSample(phsp,phsp.unfiltered_sample(10000,maximum=1.0))

sgma1 = phsp.m2ab(smp)
sgma2 = phsp.m2ac(smp)
sgma3 = phsp.m2bc(smp)
#

def decide_L_S_par(l0,l12,s0,s1,s2,p0,p1,p2):
    return  l0 > abs()

def calculate():
    pass
    # initial frame:
    #   sum all matrix elements for each sub deacay consisting of A + c, B + a and C + b
    #   for that take the heilicities of daughter particles into account (H-values as funtion of the helicities of the daughters)
    #   inside the frames of A, B and C respectively:
    #       get variables into the respective frame
    #       sum over all two body decay amplitudes of resonances going to the other two particles

def winkel_verteillung(phi,theta,J,m1,m2):
    return wigner_capital_d(phi,theta,0,J,m1,m2)

def angular_distribution_multiple_channels(phi,theta,J,s1,s2,l1,l2,bls):
    return atfi.cast_complex(helicity_couplings_from_ls(J,s1,s2,l1,l2,bls)) * atfi.cast_complex(wigner_capital_d(phi,theta,0,J,l1,l2))

phi, theta = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi,100))
phi, theta = tf.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi,100))#print(phi)
#ampl = winkel_verteillung_multiple_channels(phi,theta,sp.SPIN_2,sp.SPIN_HALF,sp.SPIN_HALF,1,1,{(sp.SPIN_1,sp.SPIN_1):1,(0,sp.SPIN_1):1})


def helicity_options(J,s1,s2):
    options = []
    for m1 in range(-s1,s1+1,2):
        for m2 in range(-s2,s2+1,2):
            if m1+m2 < J:
                options.append((m1,m2))
    return options

def coupling_options(J,s1,s2):
    bls = {}
    s_max,s_min = s1+s2, abs(s1-s2)
    for s in range(s_min,s_max+1,2):
        for l in range(0,J+s+1,2):
            if J <= l+s and J >= abs(l-s):
                bls[(l,s)] = 1
    return bls


J = sp.SPIN_2
s1 = sp.SPIN_1
s2 = sp.SPIN_1

bls = {(sp.SPIN_1,sp.SPIN_1):1,(sp.SPIN_2,sp.SPIN_0):1}

options = helicity_options(J,s1,s2)
l1,l2 = options.pop(0)
bls = coupling_options(J,s1,s2)
ampl = angular_distribution_multiple_channels(phi,theta,J,s1,s2,l1,l2,bls)
print(bls)
print(options)
for l1,l2 in options:
    ampl += angular_distribution_multiple_channels(phi,theta,J,s1,s2,l1,l2,bls)

#print(ampl)
plt.imshow(abs(ampl), extent=[0,2*np.pi,0,np.pi], origin='lower')
plt.show()


