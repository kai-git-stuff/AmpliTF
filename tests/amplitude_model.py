from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
from amplitf.dynamics import breit_wigner_decay_lineshape
from amplitf.kinematics import *
from amplitf.dalitz_decomposition import *
from amplitf.interface import sqrt
import amplitf.interface as atfi
import matplotlib.pyplot as plt
from amplitf.constants import spin as sp
from matplotlib.colors import LogNorm



def angular_distribution_multiple_channels_D(phi,theta,J,s1,s2,l1,l2,bls):
    return atfi.cast_complex(helicity_couplings_from_ls(J,s1,s2,l1,l2,bls)) * atfi.cast_complex(wigner_capital_d(phi,theta,0,J,l1,l2))

def helicity_options(J,s1,s2,s3):
    options = []
    for m1 in range(-s1,s1+1,2):
        for m2 in range(-s2,s2+1,2):
            for m3 in range(-s3,s3+1,2):
                if m1+m2+m3 <= J:
                    options.append((m1,m2,m3))
    return options

def coupling_options(J,s1,s2,P,p1,p2) -> dict:
    bls = {}
    #spins must fit
    if (sp.is_half(J) and sp.is_half(s1+s2) ) or not(sp.is_half(J) or sp.is_half(s1+s2)):
        s_max,s_min = s1+s2, abs(s1-s2)
        for s in range(s_min,s_max+1,2):
            for l in range(0,J+s+1,2):
                if J <= l+s and J >= abs(l-s)  and P == p1*p2*(-1)**l:
                    bls[(l,s)] = ((2*l + 1)/(2*J+1))**0.5
    return bls

def angular_distribution_multiple_channels_d(theta,J,s1,s2,l1,l2,nu,bls):
    return atfi.cast_complex(helicity_couplings_from_ls(J,s1,s2,l1,l2,bls)) * atfi.cast_complex(wigner_small_d(theta,J,nu,l1-l2))

class dalitz_decay:
    def __init__(self,md,ma,mb,mc,sd,sa,sb,sc,pd,pa,pb,pc):
        self.pd = pd 
        self.pa = pa 
        self.pb = pb 
        self.pc = pc 
        self.sd = sd
        self.sa = sa
        self.sb = sb
        self.sc = sc

        self.ma = ma
        self.mb = mb 
        self.mc = mc 
        self.md = md 

    def chain3(self,smp:PhaseSpaceSample,ld,la,lb,lc,resonances):
        # channel 3   
        # d -> A c : A -> a b
        sgma3 = phsp.m2ab(smp)
        sgma2 = phsp.m2ac(smp)
        sgma1 = phsp.m2bc(smp)
        ampl = atfi.zeros_tensor(sgma1.shape,atfi.ctype())
        
        # Rotation in the isobar system
        # angle between momentum of L_b and spectator 
        theta_hat = atfi.acos(cos_theta_hat_3_canonical_1(self.md, self.ma, self.mb, self.mc, sgma1, sgma2, sgma3))
        theta = atfi.acos(cos_theta_12(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        zeta_1 = atfi.acos(cos_zeta_1_aligned_3_in_frame_1(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        zeta_2 = atfi.acos(cos_zeta_2_aligned_2_in_frame_3(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        # remember to apply (-1)**((lb - lb_)/2) in front of the d matrix (switch last 2 indices)
        zeta_3 = 0
        # aligned system

        for sA,pA,helicities_A,bls_in,bls_out,X in resonances:
            for lA in helicities_A:
                bls = coupling_options(self.sd,sA,self.sc,self.pd,self.pc,pA)
                bls.update(coupling_options(self.sd,sA,self.sc,self.pd * (-1),self.pc,pA))

                H_A_c = phasespace_factor(self.md,sgma3,self.mc)* angular_distribution_multiple_channels_d(theta_hat,self.sd,sA,self.sc,lA,lc,ld,bls_in())
                
                helicities_abc = helicity_options(sA,self.sa,self.sb,self.sc)
                for la_,lb_,lc_ in helicities_abc:
                    # Rotation in the isobar system
                    # angle between A momentum (isobar) and lmbda_c in rest frame of Isobar 
                    H_a_b = phasespace_factor(sgma3,self.ma,self.mb) * angular_distribution_multiple_channels_d(theta,sA,self.sa,self.sb,la_,lb_,lA,bls_out(sgma3))
                    ampl += H_A_c * H_a_b * atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la,la_)) * atfi.cast_complex(wigner_small_d(zeta_2,self.sb,lb,lb_)) * (-1)**((lb - lb_)/2) * atfi.cast_complex(wigner_small_d(zeta_3,self.sc,lc,lc_))
        return ampl

    def chain2(self,smp:PhaseSpaceSample,ld,la,lb,lc,resonances):
        # channel 2
        # L_b -> B b : B -> (a,c)
        sgma3 = phsp.m2ab(smp)
        sgma2 = phsp.m2ac(smp)
        sgma1 = phsp.m2bc(smp)
        ampl = atfi.zeros_tensor(phsp.m2ab(smp).shape,atfi.ctype())

        zeta_1 = atfi.acos(cos_zeta_1_aligned_1_in_frame_2(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        # remember to apply (-1)**((la - la_)/2) in front of the d matrix (switch last 2 indices)
        zeta_2 = 0 # allways one is 0
        zeta_3 = atfi.acos(cos_zeta_3_aligned_2_in_frame_3(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        theta_hat =  atfi.acos(cos_theta_hat_1_canonical_2(self.md, self.ma, self.mb, self.mc, sgma1, sgma2, sgma3))
        # remember factor of (-1)**((lB - ld)/2) because we switched indices 1 and 2
        theta = atfi.acos(cos_theta_31(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))

        for sB,pB,helicities_C,bls_in,bls_out,X in resonances:
            for lB in helicities_C:
                # channel 2
                # L_b -> B b : B -> (a,c)
                H_A_c =  phasespace_factor(self.md,sgma2,self.mb)* angular_distribution_multiple_channels_d(theta_hat,self.sd,sB,self.sb,lB,lb,ld,bls_in())
                helicities_abc = helicity_options(sB,self.sa,self.sb,self.sc)
                for la_,lb_,lc_ in helicities_abc:
                    #  A -> lambda_c Dbar
                    # Rotation in the isobar system
                    H_a_b =  phasespace_factor(sgma2,self.ma,self.mc)* angular_distribution_multiple_channels_d(theta,sB,self.sa,self.sc,la_,lc_,lB,bls_out(sgma2))
                    #H_a_b = get_helicity(helicities_dict,la_,lc_,pB,pa,pc,sB,sa,sc)
                    # symmetry of the d matrices
                    H_a_b *= (-1)**((lB - ld)/2)  * (-1)**((la - la_)/2) * atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la,la_)) *  atfi.cast_complex(wigner_small_d(zeta_3,self.sc,lc,lc_))
                    ampl += H_A_c * H_a_b * atfi.cast_complex(wigner_small_d(zeta_2,self.sb,lb,lb_)) 
        return ampl

    def chain1(self,smp:PhaseSpaceSample,ld,la,lb,lc,resonances):
        # channel 3
        # L_b -> C a : C -> (b,c)  
        sgma3 = phsp.m2ab(smp)
        sgma2 = phsp.m2ac(smp)
        sgma1 = phsp.m2bc(smp)
        ampl = atfi.zeros_tensor(phsp.m2ab(smp).shape,atfi.ctype())
        # Rotation in the isobar system
        theta_hat =  atfi.acos(cos_theta_hat_1_canonical_1(self.md, self.ma, self.mb, self.mc, sgma1, sgma2, sgma3))
        # will just return one, as we are in the alligned system anyways
        theta = atfi.acos(cos_theta_23(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        zeta_1 = 0
        # own system
        zeta_2 = atfi.acos(cos_zeta_2_aligned_1_in_frame_2(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        zeta_3 = atfi.acos(cos_zeta_3_aligned_3_in_frame_1(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        # remember to apply (-1)**((lc - lc_)/2) in front of the d matrix (switch last 2 indices)
        for sC,pC,helicities_B,bls_in,bls_out,X in resonances:
            for lC in helicities_B:
                # first decay is weak -> we need all couplings even if parity is not conserved
                # we can simulate this by just merging both dicts for p = 1 and p = -1
                bls = coupling_options(self.sd,sC,self.sa,self.pd,self.pa,pC)
                bls.update(coupling_options(self.sd,sC,self.sa,self.pd * (-1),self.pa,pC))

                H_A_c = phasespace_factor(self.md,sgma1,self.ma) * angular_distribution_multiple_channels_d(theta_hat,self.sd,sC,self.sa,lC,la,ld,bls_in())

                helicities_abc = helicity_options(sC,self.sa,self.sb,self.sc)
                for la_,lb_,lc_ in helicities_abc:
                    # C -> b c
                    # Rotation in the isobar system
                    # angle between A momentum (isobar) and lmbda_c in rest frame of Isobar 
                    H_b_c = phasespace_factor(sgma1,self.mb,self.mc) * angular_distribution_multiple_channels_d(theta,sC,self.sb,self.sc,lb_,lc_,lC,bls_out(sgma1))
                    # symmetry of the d matrices
                    H_b_c *=  (-1)**((lc - lc_)/2) * atfi.cast_complex(wigner_small_d(zeta_3,self.sc,lc,lc_)) * atfi.cast_complex(wigner_small_d(zeta_2,self.sb,lb,lb_)) 
                    ampl += H_A_c * H_b_c * atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la,la_)) 

        return ampl

def get_helicity(helicities_dict:dict,l1_,l2_,P,p1,p2,J,s1,s2):
    H = helicities_dict.get((l1_,l2_),None)
    if H is None:
        return helicities_dict.get((-l1_,-l2_),None) * P*p1*p2* (-1)**(J -s1-s2)

def phasespace_factor(md,ma,mb):
    return atfi.cast_complex(4 * atfi.pi()* atfi.sqrt(md/two_body_momentum(md,ma,mb)))


class BaseResonance:
    def __init__(self,S,P,bls_in : dict, bls_out :dict):
        self._bls_in = bls_in
        self._bls_out = bls_out
        self.S,self.P = S,P

    def bls_out(self,s=None):
        bls = self._bls_out
        if s is not None:
            bls = {LS : b * self.X(s,LS[0]) for LS, b in bls.items()}
        return bls

    def bls_in(self,s=None):
        # returns 
        bls = self._bls_in
        if s is not None:
            bls = {LS : b * self.X(s,LS[0]) for LS, b in bls.items()}
        return bls

    def __iter__(self):
        return iter((self.S,self.P,self.helicities,self.bls_in,self.bls_out,self.X))
    
    @property
    def helicities(self):
        h = []
        for s in range(-self.S,self.S+1,2):
            h.append(s)
        return h
    
    def X(self,x,L):
        raise NotImplementedError("This is a base class! Do not try to use it for a resonance!")

class BWresonance(BaseResonance):
    # simple wrapper class
    def __init__(self,S,P,m0,gamma0,bls_in : dict, bls_out :dict,ma,mb):
        self.m0 = m0
        self.gamma0 = gamma0
        self.masses = (ma,mb)
        super().__init__(S,P,bls_in,bls_out)
    
    def X(self,s,L):
        # L will be given doubled, but bw need it normal
        return breit_wigner_decay_lineshape(s,self.m0,self.gamma0,self.masses[0],self.masses[1],1,L/2)
    
class KmatChannel:
    def __init__(self, m1,m2,bg):
        self.masses = m1,m2
        self.background = bg

class KmatResonance(BaseResonance):
    def __init__(self,M,couplings_out):
        self.couplings_out = couplings_out
        self._M = M
        self._M2 = M**2
        

    def coupling(self,a):
        return self.couplings_out[a]
    
    @property
    def M(self):
        return self._M
    
    @property
    def M2(self):
        return self._M2


class kmatrix(BaseResonance):
    def __init__(self,S,P,alphas,channels:list,resonances:list,bls_in,bls_out ,out_channel = 0):
        self.alphas = alphas # couplings of channel to resonance
        self.channels = channels # list of channels
        self.resonances = resonances # list of contributing resonances
        self._D = None # D matrix in storage to prevent us from computing it over and over, if it is not needed
        self._s = None # stored CMS energy, so we dont have to compute D all the time
        self.out_channel = out_channel # if the lineshape funktion is called, this is the channel we assume we want the lineshape for
        super().__init__(S,P,bls_in,bls_out)

    def get_m(self,a):
        return self.channels[a].masses

    def q(self,s,a):
        m1,m2 = self.get_m(a)
        s_a = m1 + m2
        d_a = m1-m2
        return atfi.sqrt(atfi.cast_complex((s-s_a**2) * (s-d_a**2)/(4*s) ))

    def gamma(self,s,a,L):
        return self.q(s,a)**L

    def phaseSpaceFactor(self,s,a):
        return atfi.complex(atfi.const(1/(16* atfi.pi()) * 2), atfi.const(0))* self.q(s,a)/atfi.cast_complex(atfi.sqrt(s))

    def V(self,s,a,b):
        # R = resonance index
        # a,b = channel indices
        return atfi.cast_complex(-sum((res.coupling(a) * res.coupling(b))/(s-res.M2) for res in self.resonances))

    def Sigma(self,s,a,L):
        sigma = self.phaseSpaceFactor(s,a) * self.gamma(s,a,L)**2
        return atfi.complex(atfi.const(0),atfi.const(1))*atfi.cast_complex(sigma)

    def build_D(self,s,L):
        v = []
        # we calculate v directly  as 1 - v * Sigma
        for a in range(len(self.channels)):
            v.append(list())
            for b in range(len(self.channels)):
                if a == b:
                    v[a].append(1-self.V(s,a,b)*self.Sigma(s,b,L))
                else:
                    v[a].append(-self.V(s,a,b)*self.Sigma(s,b,L))
                
        v = atfi.convert_to_tensor(v,dtype=atfi.ctype())
        shape = [-1 for _ in range(len(s.shape))] + [len(self.channels),len(self.channels)]
        v = atfi.reshape(v,shape)
        self._D = atfi.linalg_inv(v)


    def g(self,n,b):
        return self.resonances[n].coupling(b)
    
    def alpha(self,n):
        return self.alphas[n]

    def D(self,s,a,b,L):
        if s != self._s:
            self.build_D(s,L)
        return self._D[...,a,b]
    
    def P_func(self,s,b):
        p  = self.channels[b].background - sum( (res.coupling(b) * alpha )/atfi.cast_complex(s-res.M2)   for res,alpha in zip(self.resonances,self.alphas))
        return p

    def A_H(self,s,a,L):
        a_h = sum(self.gamma(s,a,L) * self.D(s,a,b,L) * self.P_func(s,b) for b in range(len(self.channels)))
        return a_h

    def X(self,s,L):
        # return the Lineshape for the specific outchannel
        return self.A_H(s,self.out_channel,L)

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

    decay = dalitz_decay(md,ma,mb,mc,sd,sa,sb,sc,pd,pa,pb,pc)
    
    # we will add the different Amplitudes
    bls_ds_kmatrix_in = {(0,1):atfi.complex(atfi.const(-1.8),atfi.const(4.4)),
                         (2,1):atfi.complex(atfi.const(-7.05),atfi.const(-4.06)),
                         (2,3):atfi.complex(atfi.const(4.96),atfi.const(-4.73))}
    bls_ds_kmatrix_out = {(2,0):atfi.complex(atfi.const(-1.064),atfi.const(-0.722))}
    alphas = [atfi.complex(atfi.const(0.00272),atfi.const(-0.00715)), atfi.complex(atfi.const(-0.00111),atfi.const(0.00394))]
    g0,g1,g2,g3 = -8.73, 6.54,6.6,-3.38
    m11,m12,m21,m22 = mb,mc,2006.85,mc 
    channels = [
        KmatChannel(m11,m12,0.0135),
        KmatChannel(m21,m22,0.0867)
    ]
    resonances = [
        KmatResonance(2713.6,[g0,g1] ),  #D^*_s1(2700)
        KmatResonance(2967.1,[g2,g3] ) # ToDo find if we assigned the g values correctly #D^*_s1(2860)
    ]
    D_kma = kmatrix(sp.SPIN_1,-1,alphas,channels,resonances,bls_ds_kmatrix_in,bls_ds_kmatrix_out)
    masses2 = (ma,mc)

    masses1 = (mb,mc)
    resonances1 = [BWresonance(sp.SPIN_0,1,atfi.cast_real(2317),38, {(0,1):atfi.complex(atfi.const(-0.017),atfi.const(-0.1256))},bls_ds_kmatrix_out,*masses1),#D_0(2317) no specific outgoing bls given :(
                    BWresonance(sp.SPIN_2,1,atfi.cast_real(2573),16.9,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1), #D^*_s2(2573)
                    BWresonance(sp.SPIN_1,-1,atfi.cast_real(2700),122,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1), #D^*_s1(2700)
                    BWresonance(sp.SPIN_1,-1,atfi.cast_real(2860),159,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1), #D^*_s1(2860)
                    #D_kma,
                    BWresonance(sp.SPIN_3,-1,atfi.cast_real(2860),53,{(4,5):atfi.complex(atfi.const(0.32),atfi.const(-0.33))},
                                                                                {(6,0):atfi.complex(atfi.const(-0.036),atfi.const(0.015))},*masses1), #D^*_s3(2860)
                    ]  
    resonances2 = [BWresonance(sp.SPIN_HALF,-1,atfi.cast_real(2791.9),8.9,{(0,1):atfi.complex(atfi.const(-0.53),atfi.const(0.69))},
                                {(0,1):atfi.complex(atfi.const(-0.0149),atfi.const(-0.0259))},*masses2), # xi_c (2790)
                    BWresonance(sp.SPIN_3HALF,-1,atfi.cast_real(2815), 2.43,{},{},*masses2)] # xi_c (2815) no bls couplings given :(

    ampl = sum(abs(decay.chain3(smp,ld,la,0,0,[]) + decay.chain2(smp,ld,la,0,0,resonances2) + decay.chain1(smp,ld,la,0,0,resonances1))**2 for la in range(-1,2,2) for ld in [-1])

    return ampl

ma = 2856.1 # lambda_c spin = 0.5 parity = 1
mb = 1864.84 # D^0 bar spin = 0 partiy = -1
mc = 493.677 # K-  spin = 0 parity = -1
md = 5912.19  # lambda_b  spin = 0.5 parity = +1
phsp = DalitzPhaseSpace(ma,mb,mc,md) 

smp = PhaseSpaceSample(phsp,phsp.rectangular_grid_sample(200, 200, space_to_sample="DP"))

#ampl = abs(three_body_decay_Daliz_plot_function(smp,phsp))**2
ampl = three_body_decay_Daliz_plot_function(smp,phsp)

sgma3 = phsp.m2ab(smp) # lmbda_c , D_bar
sgma2 = phsp.m2ac(smp) # lmbda_c , k
sgma1 = phsp.m2bc(smp) # D_bar , k
s1_name = r"$M^2(K^-,\bar{D}^0)$ in GeV$^2$"
s2_name = r"$M^2(\Lambda_c^+,K^-)$ in GeV$^2$"
s3_name = r"$M^2(\Lambda_c^+,\bar{D}^0)$ in GeV$^2$"
print(ampl,max(ampl))
my_cmap = plt.get_cmap('hot')
#rnd = atfi.random_uniform(sgma1.shape, (2, 3), minval=min(ampl), maxval=max(ampl), dtype=tf.dtypes.float64,alg='auto_select')
#mask = ampl > rnd
plt.style.use('dark_background')
plt.xlabel(s2_name)
plt.ylabel(s3_name)
plt.scatter(sgma2/1e6,sgma3/1e6,cmap=my_cmap,s=2,c=ampl,marker="s") # c=abs(ampl[mask])
plt.colorbar()
plt.savefig("Dalitz.png",dpi=400)
plt.show()
plt.close('all')
for s,name,label in zip([sgma1,sgma2,sgma3],["_D+K","L_c+K","L_c+D"],[s1_name,s2_name,s3_name]):
    plt.hist(s**0.5/1e3,weights=ampl,bins=100)
    plt.xlabel(r""+label.replace("^2","")[:-2])
    plt.savefig("Dalitz_%s.png"%name,dpi = 400)
    plt.show()
    plt.close('all')
#plt.hist2d(sgma1,sgma2,weights=ampl,bins=90,cmap=my_cmap)
