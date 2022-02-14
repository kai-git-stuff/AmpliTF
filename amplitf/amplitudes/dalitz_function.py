from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
from amplitf.dynamics import breit_wigner_decay_lineshape
from amplitf.kinematics import *
from amplitf.dalitz_decomposition import *
import amplitf.interface as atfi
from amplitf.constants import spin as sp


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

def possible_LS_states(J,s1,s2,P,p1,p2,parity_conservation=True) -> dict:
    bls = {}
    if (sp.is_half(J) and sp.is_half(s1+s2) ) or not(sp.is_half(J) or sp.is_half(s1+s2)) or not parity_conservation:
        s_max,s_min = s1+s2, abs(s1-s2)
        for s in range(s_min,s_max+1,2):
            for l in range(0,J+s+1,2):
                if J <= l+s and J >= abs(l-s)  and P == p1*p2*(-1)**l:
                    bls[(l,s)] = ((2*l + 1)/(2*J+1))**0.5
    return bls

def phasespace_factor(md,ma,mb):
    # phasespace factor for the dalitz functions
    return atfi.cast_complex(4 * atfi.pi()* atfi.sqrt(md/two_body_momentum(md,ma,mb)))

def angular_distribution_multiple_channels_d(theta,J,s1,s2,l1,l2,nu,bls):
    # ToDo why -l2 ??????? maybe the axis along which things are defined is wierd?
    # possible answer: helicity is not the actual thing here, but rather spin alignment.
    # helicity in rest frame will allways be oriented along the momentum
    # this causes (in the rest frames) the quantization axes to be 180° of each other -> l2 = -l2 on l1 quantization axis
    return atfi.cast_complex(helicity_couplings_from_ls(J,s1,s2,l1,-l2,bls)) * atfi.cast_complex(wigner_small_d(theta,J,nu,l1-l2))

class dalitz_decay:
    def __init__(self,md,ma,mb,mc,sd,sa,sb,sc,pd,pa,pb,pc,phsp = None):
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

        if phsp is None:
            self.phsp = DalitzPhaseSpace(ma,mb,mc,md)
        else:
            self.phsp = phsp

    def chain3(self,smp:PhaseSpaceSample,ld,la,lb,lc,resonances):
        # channel 3   
        # d -> A c : A -> a b
        sgma3 = self.phsp.m2ab(smp)
        sgma2 = self.phsp.m2ac(smp)
        sgma1 = self.phsp.m2bc(smp)
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
                helicities_abc = helicity_options(sA,self.sa,self.sb,self.sc)
                for la_,lb_,lc_ in helicities_abc:
                    # Rotation in the isobar system
                    # angle between A momentum (isobar) and lmbda_c in rest frame of Isobar 
                    H_A_c = phasespace_factor(self.md,sgma3,self.mc)* angular_distribution_multiple_channels_d(theta_hat,self.sd,sA,self.sc,lA,lc_,ld,bls_in())
                    H_a_b = phasespace_factor(sgma3,self.ma,self.mb) * angular_distribution_multiple_channels_d(theta,sA,self.sa,self.sb,la_,lb_,lA,bls_out(sgma3))
                    H_a_b *= atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la_,la)) * atfi.cast_complex(wigner_small_d(zeta_2,self.sb,lb_,lb)) * (-1)**((lb - lb_)/2) * atfi.cast_complex(wigner_small_d(zeta_3,self.sc,lc_,lc))
                    ampl += H_A_c * H_a_b # * atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la_,la))
        return ampl

    def chain2(self,smp:PhaseSpaceSample,ld,la,lb,lc,resonances):
        # channel 2
        # L_b -> B b : B -> (a,c)
        sgma3 = self.phsp.m2ab(smp)
        sgma2 = self.phsp.m2ac(smp)
        sgma1 = self.phsp.m2bc(smp)
        ampl = atfi.zeros_tensor(self.phsp.m2ab(smp).shape,atfi.ctype())

        zeta_1 = atfi.acos(cos_zeta_1_aligned_1_in_frame_2(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        # remember to apply (-1)**((la - la_)/2) in front of the d matrix (switch last 2 indices)
        zeta_2 = 0 # allways one is 0
        zeta_3 = atfi.acos(cos_zeta_3_aligned_2_in_frame_3(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        theta_hat =  atfi.acos(cos_theta_hat_1_canonical_2(self.md, self.ma, self.mb, self.mc, sgma1, sgma2, sgma3))
        # remember factor of (-1)**((ld - ld + lb_)/2) because we switched indices 1 and 2
        theta = atfi.acos(cos_theta_31(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))

        for sB,pB,helicities_C,bls_in,bls_out,X in resonances:
            for lB in helicities_C:
                # channel 2
                # L_b -> B b : B -> (a,c)
                
                helicities_abc = helicity_options(sB,self.sa,self.sb,self.sc)
                for la_,lb_,lc_ in helicities_abc:
                    #  A -> lambda_c Dbar
                    # Rotation in the isobar system
                    H_A_c =  phasespace_factor(self.md,sgma2,self.mb)* angular_distribution_multiple_channels_d(theta_hat,self.sd,sB,self.sb,lB,lb_,ld,bls_in())

                    H_a_b =  phasespace_factor(sgma2,self.ma,self.mc)* angular_distribution_multiple_channels_d(theta,sB,self.sa,self.sc,lc_,la_,lB,bls_out(sgma2))
                    # symmetry of the d matrices
                    H_a_b *= (-1)**((ld - ld + lb_)/2)  * (-1)**((la - la_)/2) * atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la_,la)) *  atfi.cast_complex(wigner_small_d(zeta_3,self.sc,lc_,lc)) * atfi.cast_complex(wigner_small_d(zeta_2,self.sb,lb_,lb)) 
                    ampl += H_A_c * H_a_b #  * (-1)**((ld - ld + lb_)/2)  * (-1)**((la - la_)/2) * atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la_,la))
        return ampl

    def chain1(self,smp:PhaseSpaceSample,ld,la,lb,lc,resonances):
        # channel 3
        # L_b -> C a : C -> (b,c)  
        sgma3 = self.phsp.m2ab(smp)
        sgma2 = self.phsp.m2ac(smp)
        sgma1 = self.phsp.m2bc(smp)
        ampl = atfi.zeros_tensor(self.phsp.m2ab(smp).shape,atfi.ctype())
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
                helicities_abc = helicity_options(sC,self.sa,self.sb,self.sc)
                for la_,lb_,lc_ in helicities_abc:
                    # C -> b c
                    # Rotation in the isobar system
                    # angle between A momentum (isobar) and lmbda_c in rest frame of Isobar #
                    H_A_c = phasespace_factor(self.md,sgma1,self.ma) * angular_distribution_multiple_channels_d(theta_hat,self.sd,sC,self.sa,lC,-la_,ld,bls_in())

                    H_b_c = phasespace_factor(sgma1,self.mb,self.mc) * angular_distribution_multiple_channels_d(theta,sC,self.sb,self.sc,lc_,lb_,lC,bls_out(sgma1))
                    # symmetry of the d matrices
                    H_b_c *=  (-1)**((lc - lc_)/2) * atfi.cast_complex(wigner_small_d(zeta_3,self.sc,lc_,lc)) * atfi.cast_complex(wigner_small_d(zeta_2,self.sb,lb_,lb)) * atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la_,la))
                    ampl += H_A_c * H_b_c 

        return ampl

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
                    v[a].append(1-self.V(s,a,b)*self.Sigma(s,a,L))
                else:
                    v[a].append(-self.V(s,a,b)*self.Sigma(s,a,L))
                
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
