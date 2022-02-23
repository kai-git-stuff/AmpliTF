"""
This file includes different types of resonances one might want to use for fits
"""
from amplitf.dynamics import breit_wigner_decay_lineshape
import amplitf.interface as atfi
from amplitf.constants import spin as sp
import numpy as np

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
        bls = self._bls_in
        if s is not None:
            bls = {LS : b * self.X(s,LS[0]) for LS, b in bls.items()}
        return bls

    def __iter__(self):
        return iter((self.S,self.P,self.helicities,self.bls_in,self.bls_out,self.X))
    
    @property
    def helicities(self) -> list:
        return sp.direction_options(self.S)
    
    def X(self,x,L):
        raise NotImplementedError("This is a base class! Do not try to use it for a resonance!")

class BWresonance(BaseResonance):
    def __init__(self,S,P,m0,gamma0,bls_in : dict, bls_out :dict,ma,mb,d=1500):
        self.m0 = m0
        self.gamma0 = gamma0
        self.d = d
        self.masses = (ma,mb)
        super().__init__(S,P,bls_in,bls_out)
    
    def X(self,s,L):
        # L will be given doubled, but bw need it normal
        return breit_wigner_decay_lineshape(s,self.m0,self.gamma0,self.masses[0],self.masses[1],self.d,L/2)
    
class KmatChannel:
    def __init__(self, m1,m2,L,bg,index):
        self.masses = m1,m2
        self.L = L # (doubled!!!)
        # the index by which the channel is refered to
        # usually we have one specific outgoing process and look at one set of final state
        # particles. Then we want all states with those particles as one index, so for the 
        # final state partial waves we can find the proper channel with (index, L)
        self.index = index 
        self.background = bg

class KmatResonance():
    def __init__(self,M,couplings_out):
        self.couplings_out = couplings_out
        self._M = atfi.cast_complex(M)
        self._M2 = atfi.cast_complex(M**2)
        
    def coupling(self,a):
        return self.couplings_out[a]
    
    @property
    def M(self):
        return self._M
    
    @property
    def M2(self):
        return self._M2

class kmatrix(BaseResonance):
    def __init__(self,S,P,alphas,channels:list,resonances:list,bls_in,bls_out ,width_factors=None,out_channel = 0):
        self.alphas = alphas # couplings of channel to resonance
        self.channels = channels # list of channels: type = KmatChannel
        self.resonances = resonances # list of contributing resonances
        self._D = None # D matrix in storage to prevent us from computing it over and over, if it is not needed
        self._s = None # stored CMS energy, so we dont have to compute D all the time
        self.out_channel = out_channel # if the lineshape funktion is called, this is the channel we assume we want the lineshape for
        self.channel_LS = {(c.index,c.L):i for i,c in enumerate(channels)} # we have to figure out the correct channel for a decay with a given L
        if width_factors is not None:
            self.width_factors = width_factors
        else:
            self.width_factors = [atfi.complex(atfi.const(0), atfi.const(0.)) for _ in range(len(self.channels))]
        super().__init__(S,P,bls_in,bls_out)

    def get_m(self,a):
        return self.channels[a].masses

    def q(self,s,a):
        m1,m2 = self.get_m(a)
        s_a = m1 + m2
        d_a = m1-m2
        return atfi.sqrt(atfi.cast_complex((s-s_a**2) * (s-d_a**2)/(4*s) ))

    def get_channel(self,index,L):
        return self.channel_LS[(index,L)]

    def L(self,channel):
        # L is doubled, so for calculations we need L/2
        return self.channels[channel].L/2

    def gamma(self,s,a):
        return self.q(s,a)**self.L(a)

    def phaseSpaceFactor(self,s,a):
        return atfi.complex(atfi.const(1/(8* atfi.pi())), atfi.const(0))* self.q(s,a)/atfi.cast_complex(atfi.sqrt(s))

    def V(self,s,a,b):
        # R = resonance index
        # a,b = channel indices
        return atfi.cast_complex(-sum((res.coupling(a) * res.coupling(b))/(s-res.M2) for res in self.resonances))

    def Sigma(self,s,a):
        sigma = self.phaseSpaceFactor(s,a) * self.gamma(s,a)**2 
        return atfi.complex(atfi.const(0),atfi.const(1))*(atfi.cast_complex(sigma) + self.width_factors[a])

    def build_D(self,s):
        v = []
        # we calculate v directly  as 1 - v * Sigma
        # ToDo do this with tf.stack
        v = np.zeros(list(s.shape) + [len(self.channels),len(self.channels)],dtype=np.complex128)
        for a in range(len(self.channels)):
            for b in range(len(self.channels)):
                if a == b:
                    v[...,a,b] = 1-self.V(s,a,b)*self.Sigma(s,b) 
                else:
                    v[...,a,b] = -self.V(s,a,b)*self.Sigma(s,b)
        v = atfi.convert_to_tensor(v)
        self._D = atfi.linalg_inv(v)

    def g(self,n,b):
        return self.resonances[n].coupling(b)
    
    def alpha(self,n):
        return self.alphas[n]

    def D(self,s,a,b):
        if s != self._s:
            self.build_D(s)
        return self._D[...,a,b]
    
    def P_func(self,s,b):
        p  = self.channels[b].background - sum( (res.coupling(b) * alpha )/atfi.cast_complex(s-res.M2)   for res,alpha in zip(self.resonances,self.alphas))
        return p

    def A_H(self,s,a):
        a_h = self.gamma(s,a) *  sum(self.D(s,a,b) * self.P_func(s,b) for b in range(len(self.channels)))
        return a_h

    def X(self,s,L):
        # return the Lineshape for the specific outchannel
        channel_number = self.get_channel(self.out_channel,L)
        s = atfi.cast_complex(s)
        return self.A_H(s,channel_number)
