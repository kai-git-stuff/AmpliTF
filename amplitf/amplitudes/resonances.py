"""
This file includes different types of resonances one might want to use for fits
"""
from amplitf.dynamics import blatt_weisskopf_ff, blatt_weisskopf_ff_squared, breit_wigner_lineshape, relativistic_breit_wigner, mass_dependent_width, orbital_barrier_factor
import amplitf.interface as atfi
from amplitf.kinematics import two_body_momentum
from amplitf.constants import spin as sp, angular as angular_constant
import numpy as np

class BaseResonance:
    def __init__(self,S,P,bls_in : dict, bls_out :dict):
        self._bls_in = bls_in
        self._bls_out = bls_out
        self.S,self.P = S,P

    @property
    def masses(self):
        raise NotImplementedError("This should return the masses of the daughters (for a scpecified channel if needed!)")

    @property
    def M0(self):
        raise NotImplementedError("This is the peak mass or the mean of the pole masses! It is used in the standard BLS versions!")

    def bls_out(self,s=None,d=None):
        """WARNING: do not use d != None, if the Blatt-Weisskopf FF are already used in the resonance function!"""
        bls = self._bls_out
        if s is not None:
            bls = {LS : b * self.X(s,LS[0]) for LS, b in bls.items()}
        if d is not None and s is not None:
            q = two_body_momentum(s,*self.masses)
            q0 = two_body_momentum(self.M0,*self.masses)
            bls = {LS : b * atfi.cast_complex(blatt_weisskopf_ff(q, q0, d, LS[0]) * orbital_barrier_factor(q, q0, LS[0])) for LS, b in bls.items()}
        print(bls)
        return bls

    def bls_in(self,s=None, d = None,md = None,mbachelor = None):
        bls = self._bls_in
        if s is not None:
            bls = {LS : b * self.X(s,LS[0]) for LS, b in bls.items()}
        if d is not None and s is not None and md is not None and mbachelor is not None:
            q = two_body_momentum(md,s,mbachelor)   # this is the momentum of the isobar in the main decayings particle rest frame (particle d)
            q0 = two_body_momentum(md,self.M0,mbachelor) # todo this might be wrong: we are allways at L_b resonance peak, so the BW_FF do not make sense here
            bls = {LS : b * atfi.cast_complex(blatt_weisskopf_ff(q, q0, d, LS[0]) * orbital_barrier_factor(q, q0, LS[0])) for LS, b in bls.items()}
        print(bls)
        return bls

    def __iter__(self):
        return iter((self.S,self.P,self.helicities,self.bls_in,self.bls_out,self.X))
    
    @property
    def helicities(self) -> list:
        return sp.direction_options(self.S)
    
    def X(self,x,L):
        raise NotImplementedError("This is a base class! Do not try to use it for a resonance!")

class BWresonance(BaseResonance):
    def __init__(self,S,P,m0,gamma0,bls_in : dict, bls_out :dict,ma,mb,d=5./1000.):
        self.m0 = m0
        self.gamma0 = gamma0
        self.d = d
        self._masses = (ma,mb)
        super().__init__(S,P,bls_in,bls_out)

    @property
    def masses(self):
        return self._masses

    @property
    def M0(self):
        return self.m0

    def X(self,s,L):
        # L will be given doubled, but bw need it normal
        L = L/2
        ma,mb = self.masses
        m = atfi.sqrt(s)
        p = two_body_momentum(m, ma, mb)
        p0 = atfi.cast_real(two_body_momentum(self.m0, ma , mb))
        ffr = blatt_weisskopf_ff(p, p0, self.d, L)
        width = mass_dependent_width(m, self.m0, self.gamma0, p, p0, ffr, L)
        return relativistic_breit_wigner(s,self.m0, width)# * orbital_barrier_factor(p, p0, L) * ffr
    
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

class KmatPole:
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
    def __init__(self,S,P,d,alphas,channels:list,resonances:list,bls_in,bls_out ,width_factors=None,out_channel = 0):
        self.alphas = alphas # couplings of channel to resonance
        self.channels = channels # list of channels: type = KmatChannel
        self.resonances = resonances # list of contributing poles (resonances)
        self._D = None # D matrix in storage to prevent us from computing it over and over, if it is not needed
        self.out_channel = out_channel # if the lineshape funktion is called, this is the channel we assume we want the lineshape for
        self.channel_LS = {(c.index,c.L):i for i,c in enumerate(channels)} # we have to figure out the correct channel for a decay with a given L
        if width_factors is not None:
            self.width_factors = width_factors
        else:
            self.width_factors = [atfi.complex(atfi.const(0), atfi.const(0.)) for _ in range(len(self.channels))]
        self.d = d # the momentum scale for the BWff
        super().__init__(S,P,bls_in,bls_out)

    @property
    def masses(self):
        for channel in self.channels:
            if channel.index == self.out_channel:
                return channel.masses
        raise(ValueError("No channel for given index found!"))
        return None

    @property
    def M0(self):
        return sum(res.M for res in self.resonances)/len(self.resonances)

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

    def BWF(self,s,a):
        q = self.q(s,a)
        q0 = sum(self.q(res.M2,a) for res in self.resonances)/len(self.resonances)
        blatt_weisskopf_ff(q,q0,self.d,self.L(a))


        l = self.L(a)
        def hankel1(x):
            if l == angular_constant.L_0:
                return atfi.const(1.0)
            if l == angular_constant.L_1:
                return 1 + x * x
            if l == angular_constant.L_2:
                x2 = x * x
                return 9 + x2 * (3.0 + x2)
            if l == angular_constant.L_3:
                x2 = x * x
                return 225 + x2 * (45 + x2 * (6 + x2))
            if l == angular_constant.L_4:
                x2 = x * x
                return 11025.0 + x2 * (1575.0 + x2 * (135.0 + x2 * (10.0 + x2)))
        return atfi.sqrt(1/hankel1(self.q(s,a)/1.))

    def gamma(self,s,a):
        #return 1
        return (self.q(s,a)/1.)**self.L(a) * self.BWF(s,a)

    def phaseSpaceFactor(self,s,a):
        return atfi.complex(atfi.const(1/(8* atfi.pi())), atfi.const(0))* self.q(s,a)/atfi.cast_complex(atfi.sqrt(s))

    def V(self,s,a,b):
        # R = resonance index
        # a,b = channel indices
        return atfi.cast_complex(sum((res.coupling(a) * res.coupling(b))/(res.M2-s) for res in self.resonances))

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
        return self._D[...,a,b]
    
    def P_func(self,s,b):
        p  = self.channels[b].background + sum( (res.coupling(b) * alpha )/atfi.cast_complex(res.M2-s)   for res,alpha in zip(self.resonances,self.alphas))
        return p

    def A_H(self,s,a):
        # s: squared energy
        # a: channel number
        self.build_D(s)
        a_h = self.gamma(s,a) * sum( self.D(s,a,b) * self.P_func(s,b) for b in range(len(self.channels)))
        return a_h

    def X(self,s,L):
        # return the Lineshape for the specific outchannel
        channel_number = self.get_channel(self.out_channel,L)
        s = atfi.cast_complex(s)
        return self.A_H(s,channel_number)
