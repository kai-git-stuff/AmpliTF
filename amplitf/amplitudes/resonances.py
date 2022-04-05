"""
This file includes different types of resonances one might want to use for fits
"""
from sympy import S
from amplitf.dynamics import blatt_weisskopf_ff, blatt_weisskopf_ff_squared, breit_wigner_lineshape, relativistic_breit_wigner, mass_dependent_width, orbital_barrier_factor
import amplitf.interface as atfi
from amplitf.kinematics import two_body_momentum, two_body_momentum_squared,two_body_momentum_no_tf
from amplitf.constants import spin as sp, angular as angular_constant
import numpy as np

class BaseResonance:
    def __init__(self,S,P,bls_in : dict, bls_out :dict, d = None):
        self._bls_in = bls_in
        self._bls_out = bls_out
        self.S,self.P = S,P
        self.d = atfi.cast_real(d)   # resonance radius (if None)


    def update(self,S,P,bls_in : dict, bls_out :dict, d = None):
        self._bls_in = bls_in
        self._bls_out = bls_out
        self.S,self.P = S,P
        self.d = atfi.cast_real(d)   # resonance radius (if None)

    @property
    def masses(self):
        raise NotImplementedError("This should return the masses of the daughters (for a scpecified channel if needed!)")

    @property
    def M0(self):
        raise NotImplementedError("This is the peak mass or the mean of the pole masses! It is meant for use in the standard BLS versions! It must be implemented!")

    @property
    def p0(self):
        return two_body_momentum(atfi.const(self.M0),*self.masses)

    def breakup_momentum(self,md,s,mbachelor):
        return two_body_momentum(md,s,mbachelor)

    def bls_out(self,s=None,d=None):
        """WARNING: do not use d != None, if the Blatt-Weisskopf FF are already used in the resonance function!"""
        bls = self._bls_out
        if s is not None:
            q = self.breakup_momentum(s,*self.masses)
            q0 = self.p0
            bls = {LS : b * self.X(s,LS[0]) * 
                    orbital_barrier_factor(atfi.cast_complex(q), atfi.cast_complex(q0), LS[0]/2) 
                    for LS, b in bls.items()}
        if self.d is not None and s is not None:
            q = self.breakup_momentum(s,*self.masses)
            q0 = self.p0
            bls = {LS : b * blatt_weisskopf_ff(q, q0, self.d, atfi.const(LS[0]/2))  for LS, b in bls.items()}
        return bls

    def bls_in(self,s=None, d = None,md = None,mbachelor = None):
        bls = self._bls_in
        if s is not None and md is not None and mbachelor is not None:
            q = self.breakup_momentum(atfi.const(md),s,atfi.const(mbachelor))   # this is the momentum of the isobar in the main decayings particle rest frame (particle d)
            q0 = two_body_momentum(atfi.const(md),self.M0,atfi.const(mbachelor)) # todo this might be wrong: we are allways at L_b resonance peak, so the BW_FF do not make sense here
            if d is not None:
                bls = {LS : b * blatt_weisskopf_ff(q, q0, d, atfi.const(LS[0]/2)) * 
                        atfi.cast_complex(orbital_barrier_factor(atfi.cast_complex(q), atfi.cast_complex(q0), atfi.const(LS[0]/2)))
                    for LS, b in bls.items()}
            else:
                bls = {LS : b * atfi.cast_complex(orbital_barrier_factor(atfi.cast_complex(q), atfi.cast_complex(q0), atfi.const(LS[0]/2)))
                    for LS, b in bls.items()}
        return bls

    def __iter__(self):
        return iter((self.S,self.P,self.helicities,self._X,self.M0,self.d,self.p0))
    
    @property
    def helicities(self) -> list:
        return sp.direction_options(self.S)
    
    def X(self,x,L):
        raise NotImplementedError("This is a base class! Do not try to use it for a resonance!")

    def __ne__(self, other):
        raise NotImplementedError("Please implement, so the Tree fitter still works")

class BWresonance(BaseResonance):
    def __init__(self,S,P,m0,gamma0,bls_in : dict, bls_out :dict,ma,mb,s,d=5./1000.):
        self.m0 = m0 # atfi.const(m0)
        self.gamma0 = atfi.const(gamma0)
        self._masses = (atfi.const(ma),atfi.const(mb))
        self._p0 = two_body_momentum(self.M0, *self._masses)
        self._X = None
        self._X = self.X(s) # angular momentum not relevant for pure lineshape
        super().__init__(S,P,bls_in,bls_out,d)

    def update(self,S,P,m0,gamma0,bls_in : dict, bls_out :dict,ma,mb,d=5./1000.):
        self.m0 = m0 # atfi.const(m0)
        self.gamma0 = gamma0
        self._masses = (atfi.const(ma),atfi.const(mb))
        self._p0 = two_body_momentum(self.M0, *self._masses)
        super().update(S,P,bls_in,bls_out,d)

    @property
    def masses(self):
        return self._masses

    @property
    def M0(self):
        return self.m0
    
    @property
    def p0(self):
        return self._p0

    @atfi.function
    def X(self,s):
        if self._X is not None:
            return self._X
        # L will be given doubled, but bw needs it normal
        # L = L/2
        # ma,mb = self.masses
        # m = atfi.sqrt(s)
        # p = two_body_momentum(m, ma, mb)
        # ffr = atfi.cast_real(blatt_weisskopf_ff(p, self.p0, self.d, atfi.const(L)))
        # width = mass_dependent_width(m, self.m0, self.gamma0, p, self.p0, ffr, L)
        return relativistic_breit_wigner(s,self.m0, self.gamma0)
 
    def __ne__(self, other):
        return self.gamma != other.gamma or self.M0 != other.M0

class subThresholdBWresonance(BWresonance):
    def __init__(self, S, P, m0, gamma0, bls_in: dict, bls_out: dict, ma, mb,mc,md,s, d=5 / 1000):
        """A variation of the BW resonance, that sits beneeth a threshold for our decay products"""
        super().__init__(S, P, m0, gamma0, bls_in, bls_out, ma, mb, s, d)
        p0_2 = two_body_momentum_squared(self.M0, ma , mb)
        self._p0 = atfi.sqrt(atfi.cast_complex(p0_2))
        self.d = atfi.cast_complex(self.d)
        self._X = None
        self._X = self.X(s)

    def breakup_momentum(self,md,s,mbachelor):
        return atfi.cast_complex(two_body_momentum(md,s,mbachelor))

    @atfi.function
    def X(self,s):
        # L will be given doubled, but bw needs it normal
        # L = L/2
        # ma,mb = self.masses
        # m = atfi.cast_complex(atfi.sqrt(s))
        # p = atfi.cast_complex(two_body_momentum(m, atfi.cast_complex(ma), atfi.cast_complex(mb)))
        # ffr = blatt_weisskopf_ff(p, self.p0, atfi.cast_complex(self.d), atfi.const(L))
        # width = mass_dependent_width(m, atfi.cast_complex(self.m0), atfi.cast_complex(self.gamma0), p, self.p0, ffr, L)
        return relativistic_breit_wigner(atfi.cast_complex(s),self.m0, atfi.cast_complex(self.gamma0))

class KmatChannel:
    def __init__(self, m1,m2,L,bg,index):
        self.masses = atfi.const(m1),atfi.const(m2)
        self.L = L # (doubled!!!)
        # the index by which the channel is refered to
        # usually we have one specific outgoing process and look at one set of final state
        # particles. Then we want all states with those particles as one index, so for the 
        # final state partial waves we can find the proper channel with (index, L)
        self.index = index 
        self.background = bg
    
    def update(self, m1,m2,L,bg,index):
        self.masses = atfi.const(m1),atfi.const(m2)
        self.L = L # (doubled!!!)
        # the index by which the channel is refered to
        # usually we have one specific outgoing process and look at one set of final state
        # particles. Then we want all states with those particles as one index, so for the 
        # final state partial waves we can find the proper channel with (index, L)
        self.index = index 
        self.background = bg

class KmatPole:
    def __init__(self,M,couplings_out:list):
        self.couplings_out = couplings_out
        self._M = atfi.complex(atfi.const(M),atfi.const(0))
        self._M2 = atfi.complex(atfi.const(M**2),atfi.const(0))
        
    def coupling(self,a):
        return self.couplings_out[a]
    
    def update(self,M,couplings_out:list):
        self.couplings_out = couplings_out
        self._M = atfi.complex(atfi.const(M),atfi.const(0))
        self._M2 = atfi.complex(atfi.const(M**2),atfi.const(0))
    
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
            self.width_factors = [atfi.complex(atfi.const(0.), atfi.const(0.)) for _ in range(len(self.channels))]
        self.d = d # the momentum scale for the BWff
        self._p0 = two_body_momentum(self.M0, *self.masses)
        super().__init__(S,P,bls_in,bls_out,d)

    @property
    def masses(self):
        for channel in self.channels:
            if channel.index == self.out_channel:
                return channel.masses
        raise(ValueError("No channel for given index %s found!"%self.out_channel))
        return None

    @property
    def M0(self):
        """Mean of pole positions"""
        return atfi.cast_real(sum(res.M for res in self.resonances)/len(self.resonances))

    @property
    def p0(self):
        return self._p0
    @atfi.function
    def get_m(self,a):
        return self.channels[a].masses
    @atfi.function
    def q(self,s,a):
        m1,m2 = self.get_m(a)
        # return atfi.cast_complex(two_body_momentum(s,atfi.cast_complex(m1),atfi.cast_complex(m2)))
        s_a = atfi.cast_complex(m1 + m2)
        d_a = atfi.cast_complex(m1-m2)
        return atfi.sqrt(atfi.cast_complex((s-s_a**2) * (s-d_a**2)/(4*s) ))
    @atfi.function
    def get_channel(self,index,L):
        return self.channel_LS[(index,L)]
    @atfi.function
    def L(self,channel):
        # L is doubled, so for calculations we need L/2
        return atfi.cast_complex(atfi.const(self.channels[channel].L/2))
    @atfi.function
    def BWF(self,s,a):
        q = self.q(s,a)
        q0 = self.q(self.M0**2,a)
        return blatt_weisskopf_ff(q,q0,self.d,atfi.const(self.L(a)))
    @atfi.function
    def gamma(self,s,a):
        return (self.q(s,a))**self.L(a) 
    @atfi.function
    def phaseSpaceFactor(self,s,a):
        return atfi.complex(1/(8* atfi.pi()), atfi.const(0))* self.q(s,a)/atfi.cast_complex(atfi.sqrt(s))
    @atfi.function
    def V(self,s,a,b):
        # R = resonance index
        # a,b = channel indices
        return atfi.cast_complex(sum((res.coupling(a) * res.coupling(b))/(res.M2-s) for res in self.resonances))
    @atfi.function
    def Sigma(self,s,a):
        sigma = self.phaseSpaceFactor(s,a) * self.gamma(s,a)**2 
        return atfi.complex(atfi.const(0),atfi.const(1.))*(atfi.cast_complex(sigma) + self.width_factors[a])
    @atfi.function
    def build_D(self,s):
        # we calculate v directly  as 1 - v * Sigma
        # ToDo do this with tf.stack
        # v = atfi.zeros(list(s.shape) + [len(self.channels),len(self.channels)] )
        v = list()
        for a in range(len(self.channels)):
            temp = list()
            for b in range(len(self.channels)):
                if a == b:
                    # temp.append(atfi.ones(s) * b)
                    temp.append(1 -self.V(s,a,b)*self.Sigma(s,b) )
                else:
                    # temp.append(atfi.ones(s) * b)
                    temp.append(-self.V(s,a,b)*self.Sigma(s,b))
            v.append(atfi.stack(temp,-1))
        v = atfi.stack(v,-2)
        D = atfi.linalg_inv(v)
        return D
    @atfi.function
    def g(self,n,b):
        return self.resonances[n].coupling(b)
    @atfi.function
    def alpha(self,n):
        return self.alphas[n]
    @atfi.function
    def D(self,s,a,b):            
        return self.build_D(s)[...,a,b]
    @atfi.function
    def P_func(self,s,b):
        p  = self.channels[b].background + sum( (res.coupling(b) * alpha )/atfi.cast_complex(res.M2-s)   for res,alpha in zip(self.resonances,self.alphas))
        return p
        
    @atfi.function
    def A_H(self,s,a):
        # s: squared energy
        # a: channel number
        #a_h = self.gamma(s,a) * sum( self.D(s,a,b) * self.P_func(s,b) for b in range(len(self.channels)))
        a_h = sum( self.D(s,a,b) * self.P_func(s,b) for b in range(len(self.channels))) # because The barrier factors are sourced out of the resonance lineshape
        return a_h

    @atfi.function
    def X(self,s):
        # return the Lineshape for the specific outchannel
        channel_number = self.out_channel
        s = atfi.cast_complex(s)
        self.build_D(s)
        return self.A_H(s,channel_number) 
