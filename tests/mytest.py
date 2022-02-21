from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
from amplitf.dynamics import breit_wigner_decay_lineshape
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

smp = PhaseSpaceSample(phsp,phsp.rectangular_grid_sample(200, 200, space_to_sample="DP"))

sgma1 = phsp.m2ab(smp)
sgma2 = phsp.m2ac(smp)
sgma3 = phsp.m2bc(smp)

def winkel_verteillung(phi,theta,J,m1,m2):
    return wigner_capital_d(phi,theta,0,J,m1,m2)

def helicity_coupling_times_d(phi,theta,J,s1,s2,l1,l2,bls):
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

def coupling_options(J,s1,s2,P,p1,p2) -> dict:
    bls = {}
    #spins must fit
    if sp.is_half(J) and sp.is_half(s1+s2) or not(sp.is_half(J) or sp.is_half(s1+s2)):
        s_max,s_min = s1+s2, abs(s1-s2)
        for s in range(s_min,s_max+1,2):
            for l in range(0,J+s+1,2):
                if J <= l+s and J >= abs(l-s)  and P == p1*p2*(-1)**l:
                    bls[(l,s)] = 0.03
    return bls



@atfi.function
def two_body_decay_with_fixed_J(J,s1,s2,P,p1,p2,theta):
    ampl = atfi.zeros_tensor(theta.shape,atfi.fptype())
    options = helicity_options(J,s1,s2)
    bls = coupling_options(J,s1,s2,P,p1,p2)
    print(bls)
    print(options)
    for l1,l2 in options:
        ampl += abs(helicity_coupling_times_d(phi,theta,J,s1,s2,l1,l2,bls))**2
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



def helicity_coupling_times_d(theta,J,s1,s2,l1,l2,bls):
    return atfi.cast_complex(helicity_couplings_from_ls(J,s1,s2,l1,l2,bls)) * atfi.cast_complex(wigner_small_d(theta,J,l1,l2))

def helicity_coupling_times_d(theta,J,s1,s2,l1,l2,nu,bls):
    return atfi.cast_complex(helicity_couplings_from_ls(J,s1,s2,l1,l2,bls)) * atfi.cast_complex(wigner_small_d(theta,J,nu,l1-l2))


def chain3(smp:PhaseSpaceSample,la,lb,lc):
    # channel 1
    # L_b - > A k : A -> lambda_c Dbar
    # d -> (a b) c
    k = 3 # helper index for the specified decay chain

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
    helicities_A_d = [(1,1),(1,0),(1,-1),(0,1),(0,0),(0,-1),(-1,1),(-1,0),(-1,-1)]
    helicities_ab = [(1,0),(0,0),(-1,0)]
    for lA, ld in helicities_A_d:
        # channel 1
        # L_b - > A k : A -> lambda_c Dbar

        
        if lA - ld - lc != 0:
            # spin of Kaon is 0
            # helicity is conserved in every system
            continue

        # Rotation in the isobar system
        # angle between momentum of L_b and spectator(Kaon)
        theta = atfi.acos(cos_theta_hat_3_canonical_1(md, ma, mb, mc, sgma1, sgma2, sgma3))
        
        
        # A does not have definite Spin 
        # assume A has spin half first
        # we will add the different Amplitudes
        sA = sp.SPIN_3HALF
        pA = -1
        bls = coupling_options(sd,sA,sc,pd,pc,pA)
        bls.update(coupling_options(sd,sA,sc,pd * (-1),pc,pA))

        #print(bls)
        # Kaon has spin 0, so we can ditch the sum over the kaon helicities
        H_A_c = helicity_coupling_times_d(theta,sd,sA,sc,lA,0,ld,bls)
        
        theta = atfi.acos(cos_theta_12(md,ma,mb,mc,sgma1,sgma2,sgma3))
        bls = coupling_options(sA,sa,sb,pA,pa,pb)
        m0 = atfi.cast_real(4900)
        x =0

        zeta_1 = atfi.acos(cos_zeta_1_aligned_3_in_frame_1(md,ma,mb,mc,sgma1,sgma2,sgma3))

        zeta_2 = atfi.acos(cos_zeta_2_aligned_2_in_frame_3(md,ma,mb,mc,sgma1,sgma2,sgma3))
        #remember to apply (-1)**((lb - lb_)/2) in front of the d matrix (switch last 2 indices)

        zeta_3 = 0
        # trust me

        for la_,lb_ in helicities_ab:
            #  A -> lambda_c Dbar
            # Rotation in the isobar system
            # angle between A momentum (isobar) and lmbda_c in rest frame of Isobar 
            H_a_b = helicity_coupling_times_d(theta,sA,sa,sb,la_,lb_,lA,bls)

            # no sum over lc_ needed, because sc is 0
            ampl += H_A_c * H_a_b * x * atfi.cast_complex(wigner_small_d(zeta_1,sa,la,la_)) * atfi.cast_complex(wigner_small_d(zeta_2,sb,lb,lb_)) * (-1)**((lb - lb_)/2) * atfi.cast_complex(wigner_small_d(zeta_3,sc,lc,0))
    return ampl

def chain2(smp:PhaseSpaceSample,la,lb,lc):
    # channel 2
    # L_b - > B D : B -> lambda_c k
    # L_b -> B b : B -> (a,c)
    k = 2

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

    ampl = atfi.zeros_tensor(phsp.m2ab(smp).shape,atfi.ctype())
    helicities_B_d = [(1,1),(1,0),(1,-1),(0,1),(0,0),(0,-1),(-1,1),(-1,0),(-1,-1)]
    helicities_a_c = [(1,0),(0,0),(-1,0)]

    for lB, ld in helicities_B_d:

        if lB - ld - lb != 0:
            # spin of D meson is 0
            continue
        # channel 2
        # L_b - > B D : B -> lambda_c k
        # L_b -> B b : B -> (a,c)

        # Rotation in the isobar system
        # angle between momentum of L_b and spectator(Kaon)
        theta =  atfi.acos(cos_theta_hat_1_canonical_2(md, ma, mb, mc, sgma1, sgma2, sgma3))
        
        # A does not have definite Spin 
        # assume A has spin half first
        # we will add the different Amplitudes
        sB = sp.SPIN_HALF
        pB = -1
        # first decay is weak -> we need all couplings even if parity is not conserved
        # we can simulate this by just merging both dicts for p = 1 and p = -1
        bls = coupling_options(sd,sB,sb,pd,pb,pB)
        bls.update(coupling_options(sd,sB,sb,pd * (-1),pb,pB))

        #print(bls)
        # D meson has spin 0, so we can ditch the sum over the kaon helicities
        H_A_c = helicity_coupling_times_d(theta,sd,sB,sb,lB,0,ld,bls)
        
        theta = atfi.acos(cos_theta_31(md,ma,mb,mc,sgma1,sgma2,sgma3))
        bls = coupling_options(sB,sa,sc,pB,pa,pc)
        m0 = atfi.cast_real(3500.0)
        x =0

        zeta_1 = atfi.acos(cos_zeta_1_aligned_1_in_frame_2(md,ma,mb,mc,sgma1,sgma2,sgma3))
        #remember to apply (-1)**((la - la_)/2) in front of the d matrix (switch last 2 indices)

        zeta_2 = 0 # allways one is 0

        zeta_3 = atfi.acos(cos_zeta_3_aligned_2_in_frame_3(md,ma,mb,mc,sgma1,sgma2,sgma3))

        for la_,lc_ in helicities_a_c:
            #  A -> lambda_c Dbar
            # Rotation in the isobar system
            # angle between A momentum (isobar) and lmbda_c in rest frame of Isobar 
            H_a_b = helicity_coupling_times_d(theta,sB,sa,sc,la_,lc_,lB,bls)
            
            # symmetry of the d matrices
            H_a_b *= (-1)**(lB - ld)  * (-1)**((la - la_)/2) * atfi.cast_complex(wigner_small_d(zeta_1,sa,la,la_)) *  atfi.cast_complex(wigner_small_d(zeta_3,sc,lc,lc_))


            ampl += H_A_c * H_a_b * x * atfi.cast_complex(wigner_small_d(zeta_2,sb,lb,0)) 

    return ampl

class resonance:
    # simple cover c
    def __init__(self,S,P,m0,gamma0,weight):
        self.S = S
        self.P = P
        self.m0 = m0
        self.gamma0 = gamma0
        self.weight = weight

    def __iter__(self):
        return iter((self.S,self.P,self.m0,self.gamma0,self.weight,self.helicities))
    
    def helicities(self,other_S):
        h = []
        for sa in range(-self.S,self.S+1,2):
            for sb in range(-other_S,other_S+1,2):
                h.append((sa,sb))
        return  h

def chain1(smp:PhaseSpaceSample,la,lb,lc):
    # channel 3
    # L_b - > C D : B -> D_bar k
    # L_b -> C a : B -> (b,c)
    k = 1

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

    ampl = atfi.zeros_tensor(phsp.m2ab(smp).shape,atfi.ctype())
    helicities_C_d = [(1,0),(0,0),(-1,0)]
    helicities_b_c = [(0,0)]



    # Rotation in the isobar system
    theta =  atfi.acos(cos_theta_hat_1_canonical_1(md, ma, mb, mc, sgma1, sgma2, sgma3))
    # will just return one, as we are in the alligned system anyways


    # we will add the different Amplitudes
    # first amplitude: D_0(2317)
    resonances = [resonance(sp.SPIN_0,1,atfi.cast_real(2317),30,0.099 ),
                    resonance(sp.SPIN_3,-1,atfi.cast_real(2860),53,0.0181)]
    theta = atfi.acos(cos_theta_23(md,ma,mb,mc,sgma1,sgma2,sgma3))
    zeta_1 = 0
    # own system
    zeta_2 = atfi.acos(cos_zeta_2_aligned_1_in_frame_2(md,ma,mb,mc,sgma1,sgma2,sgma3))
    zeta_3 = atfi.acos(cos_zeta_3_aligned_3_in_frame_1(md,ma,mb,mc,sgma1,sgma2,sgma3))
    #remember to apply (-1)**((lc - lc_)/2) in front of the d matrix (switch last 2 indices)
    for sC,pC,m0,gamma0,weight,helicities_C in resonances:
        for lC, ld in helicities_C(sd):
            if lC - ld - lb != 0:
                # helicites first defined in mother particle system
                continue
            # channel 1
            # L_b -> C a : C -> (b,c)
            # first decay is weak -> we need all couplings even if parity is not conserved
            # we can simulate this by just merging both dicts for p = 1 and p = -1
            bls = coupling_options(sd,sC,sa,pd,pa,pC)
            bls.update(coupling_options(sd,sC,sa,pd * (-1),pa,pC))

            # sum over initial frame lambda_c helicities is done outside of this function
            H_A_c = helicity_coupling_times_d(theta,sd,sC,sa,lC,la,ld,bls)


            x = weight* breit_wigner_decay_lineshape(sgma1,m0,gamma0,ma,mb,1,0)
            bls = coupling_options(sC,sb,sc,pC,pb,pc)
            for lb_,lc_ in helicities_b_c:
                #  C -> lambda_c k
                # C -> b c
                # Rotation in the isobar system
                # angle between A momentum (isobar) and lmbda_c in rest frame of Isobar 
                H_a_b = helicity_coupling_times_d(theta,sC,sb,sc,lb_,lc_,lC,bls)
                
                # symmetry of the d matrices
                H_a_b *=  (-1)**((lc - lc_)/2) * atfi.cast_complex(wigner_small_d(zeta_3,sc,lc,lc_)) * atfi.cast_complex(wigner_small_d(zeta_2,sb,lb,lb_)) 

                for la_ in range(-sa,sa+1):
                    # go over all possible helicities of a in the b c system with regard to the d (mother particle) system
                    ampl += H_A_c * H_a_b * x *atfi.cast_complex(wigner_small_d(zeta_1,sa,la,la_)) 

    return ampl

def three_body_decay_Daliz_plot_function(smp,phsp:DalitzPhaseSpace,**kwargs):
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

    ampl = sum(chain3(smp,la,0,0) + chain2(smp,la,0,0) + chain1(smp,la,0,0) for la in range(-1,2))

    return ampl

ampl = abs(three_body_decay_Daliz_plot_function(smp,phsp))**2
sgma3 = phsp.m2ab(smp)
sgma2 = phsp.m2ac(smp)
sgma1 = phsp.m2bc(smp)

my_cmap = plt.get_cmap('hot')
rnd = atfi.random_uniform(sgma1.shape, (2, 3), minval=min(ampl), maxval=max(ampl), dtype=tf.dtypes.float64,alg='auto_select')
#mask = ampl > rnd
plt.style.use('dark_background')

plt.scatter(sgma1,sgma2,cmap=my_cmap,s=2,c=ampl,marker="s") # c=abs(ampl[mask])
#plt.hist2d(sgma1,sgma2,weights=ampl,bins=90,cmap=my_cmap)
plt.show()