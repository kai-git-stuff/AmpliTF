from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
from amplitf.dynamics import breit_wigner_lineshape
from amplitf.kinematics import *
from amplitf.dalitz_decomposition import *
from amplitf.interface import sqrt
import amplitf.interface as atfi
import matplotlib.pyplot as plt
from amplitf.constants import spin as sp






ma = 2856.1 # lambda_c spin = 0.5 parity = 1
mb = 1864.84 # D^0 bar spin = 0 partiy = -1
mc = 493.677 # K-  spin = 0 parity = -1
md = 5912.19  # lambda_b  spin = 0.5 parity = +1
phsp = DalitzPhaseSpace(ma,mb,mc,md) 

smp = PhaseSpaceSample(phsp,phsp.uniform_sample(10000))

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

def angular_distribution_multiple_channels_D(phi,theta,J,s1,s2,l1,l2,bls):
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

def coupling_options(J,s1,s2,P,p1,p2):
    bls = {}
    #spins must fit
    if sp.is_half(J) and sp.is_half(s1+s2) or not(sp.is_half(J) or sp.is_half(s1+s2)):
        s_max,s_min = s1+s2, abs(s1-s2)
        for s in range(s_min,s_max+1,2):
            for l in range(0,J+s+1,2):
                if J <= l+s and J >= abs(l-s)  and P == p1*p2*(-1)**l:
                    bls[(l,s)] = 1
    return bls



@atfi.function
def two_body_decay_with_fixed_J(J,s1,s2,P,p1,p2,theta):
    ampl = atfi.zeros_tensor(theta.shape,atfi.fptype())
    options = helicity_options(J,s1,s2)
    bls = coupling_options(J,s1,s2,P,p1,p2)
    print(bls)
    print(options)
    for l1,l2 in options:
        ampl += abs(angular_distribution_multiple_channels_D(phi,theta,J,s1,s2,l1,l2,bls))**2
    return ampl

J = sp.SPIN_1
s1 = sp.SPIN_0
s2 = sp.SPIN_HALF
P = 1
p1 = 1
p2 = 1


ampl = atfi.zeros_tensor(phi.shape,atfi.fptype())
for J in [sp.SPIN_0,sp.SPIN_HALF,sp.SPIN_3HALF]:
    continue
    ampl += two_body_decay_with_fixed_J(J,s1,s2,P,p1,p2,theta)



def angular_distribution_multiple_channels_d(theta,J,s1,s2,l1,l2,bls):
    return atfi.cast_complex(helicity_couplings_from_ls(J,s1,s2,l1,l2,bls)) * atfi.cast_complex(wigner_small_d(theta,J,l1,l2))


def three_body_decay(smp,phsp:DalitzPhaseSpace):
    jd = sp.SPIN_HALF
    pd = 1 # lambda_b  spin = 0.5 parity = +1
    pa = 1 # lambda_c spin = 0.5 parity = 1
    pb = -1 # D^0 bar spin = 0 partiy = -1
    pc = -1 # K-  spin = 0 parity = -1
    sd = sp.SPIN_HALF
    sa = sp.SPIN_HALF
    sb = sp.SPIN_0
    sc = sp.SPIN_0

    ma = 2856.1 # lambda_c spin = 0.5 parity = 1
    mb = 1864.84 # D^0 bar spin = 0 partiy = -1
    mc = 493.677 # K-  spin = 0 parity = -1
    md = 5912.19  # lambda_b  spin = 0.5 parity = +1
    
    sgma3 = phsp.m2ab(smp)
    sgma2 = phsp.m2ac(smp)
    sgma1 = phsp.m2bc(smp)

    # channel 1
    # L_b - > 
    ampl = atfi.zeros_tensor(phsp.m2ab(smp).shape,atfi.ctype())
    helicities_L_b_L_c = [(1,1),(1,-1),(1,0),(0,0),(0,1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    for la, ld in helicities_L_b_L_c:
        # channel 1
        # L_b - > A k : A -> lambda_c Dbar
        theta = atfi.acos(cos_theta_12(md, ma, mb, mc, sgma1, sgma2, sgma3))
        # A does not have definite Spin 
        # assume A has spin half first
        # we will add the different Amplitudes
        sA = sp.SPIN_HALF
        pA = -1
        bls = coupling_options(sd,sA,sc,P,pc,pA)
        H_A_c = angular_distribution_multiple_channels_d(theta,sd,sA,sc,ld,0,bls)
        ampl += H_A_c
    return ampl


ampl = three_body_decay(smp,phsp)
sgma3 = phsp.m2ab(smp)
sgma2 = phsp.m2ac(smp)
sgma1 = phsp.m2bc(smp)

my_cmap = plt.get_cmap('hot')
rnd = atfi.random_uniform(sgma1.shape, (2, 3), minval=0, maxval=max(abs(ampl)), dtype=tf.dtypes.float64,alg='auto_select')
mask = abs(ampl) > rnd
print(abs(ampl))
print(rnd)
print(mask)
plt.scatter(sgma1[mask],sgma2[mask],cmap=my_cmap,c=abs(ampl[mask]),s=2)

plt.show()


