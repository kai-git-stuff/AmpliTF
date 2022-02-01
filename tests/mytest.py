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





@atfi.function
def fac(n):
    if (n< 2):
        return 1
    return atfi.reduce_prod(atfi.range(1,n+1))

@atfi.function
def d(J_x,lmbda_c,lmbda_D,theta):
    s = atfi.range(max(-(lmbda_c-lmbda_D),0),max(lmbda_D + J_x, J_x - lmbda_c))
    theta_2 = theta/2.
    return (
        sqrt(fac(J_x + lmbda_c) *fac(J_x - lmbda_c) * fac(J_x + lmbda_D) *  fac(J_x - lmbda_D)) * 
    atfi.sum((-1)**s /(fac(J_x + lmbda_D - s) * fac(s) * fac(lmbda_c - lmbda_D + s) * fac(J_x - lmbda_c - s)) * 
    atfi.cos(theta_2)**(2*J_x +lmbda_D - lmbda_c - 2*s) * 
    atfi.sin(theta_2)**(lmbda_c - lmbda_D - 2*s))
    )


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

def winkel_verteillung_multiple_channels(phi,theta,J,s1,s2,l1,l2,bls):
    return atfi.cast_complex(helicity_couplings_from_ls(J,s1,s2,l1,l2,bls)) * atfi.cast_complex(wigner_capital_d(phi,theta,0,J,l1,l2))

phi, theta = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi,100))
#print(phi)
#ampl = winkel_verteillung_multiple_channels(phi,theta,sp.SPIN_2,sp.SPIN_HALF,sp.SPIN_HALF,1,1,{(sp.SPIN_1,sp.SPIN_1):1,(0,sp.SPIN_1):1})


J = sp.SPIN_2
s1 = sp.SPIN_HALF
s2 = sp.SPIN_HALF
l1 = 1
l2 = 1
bls = {(sp.SPIN_1,sp.SPIN_1):1,(sp.SPIN_2,sp.SPIN_0):1}

ampl = winkel_verteillung_multiple_channels(phi,theta,J,s1,s2,l1,l2,bls)











#print(ampl)
plt.imshow(abs(ampl), extent=[0,2*np.pi,0,np.pi], origin='lower')
plt.show()


@atfi.function
def H(lmbda_i,lmbda_f):
    return atfi.complex(1,0)

@atfi.function
def R(m,M0,gamma):
    pass


@atfi.function
def M(lmbda_i,lmbda_f,j):
    pass
