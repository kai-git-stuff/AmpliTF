from sympy import S
from amplitf.dynamics import blatt_weisskopf_ff, blatt_weisskopf_ff_squared, breit_wigner_lineshape, relativistic_breit_wigner, mass_dependent_width, orbital_barrier_factor
import amplitf.interface as atfi
from amplitf.kinematics import two_body_momentum, two_body_momentum_squared,two_body_momentum_no_tf
from amplitf.constants import spin as sp, angular as angular_constant
import numpy as np
from amplitf.amplitudes.resonances import BaseResonance
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


class KmatPole:
    def __init__(self,M,couplings_out:list):
        self.couplings_out = couplings_out
        self._M = atfi.complex(atfi.const(M),atfi.const(0))
        self._M2 = atfi.complex(atfi.const(M**2),atfi.const(0))
        

class kmatrix(BaseResonance):
    def __init__(self,S,P,d,alphas,channels:list,resonances:list,bls_in,bls_out ,width_factors=None,out_channel = 0):
        self.alphas = alphas # couplings of channel to resonance
        self.channels = channels # list of channels: type = KmatChannel
        self.resonances = resonances # list of contributing poles (resonances)
        self._D = None # D matrix in storage to prevent us from computing it over and over, if it is not needed
        self.out_channel = out_channel # if the lineshape funktion is called, this is the channel we assume we want the lineshape for
        self.channel_LS = {(c[0],c[1]):i for i,c in enumerate(channels)} # we have to figure out the correct channel for a decay with a given L
        if width_factors is not None:
            self.width_factors = width_factors
        else:
            self.width_factors = [atfi.complex(atfi.const(0.), atfi.const(0.)) for _ in range(len(self.channels))]
        self.d = d # the momentum scale for the BWff
        self._p0 = two_body_momentum(self.M0, *self.masses)
        super().__init__(S,P,bls_in,bls_out,d)
        '''channels = [(index,L,bg,m1,m2)]
        resonances = [(M,couplings)]'''
    
def masses(channels,alphas,resonances,out_channel):
    for channel in channels:
        if channel[0] == out_channel:
            return channel[3],channel[4]
    raise(ValueError("No channel for given index %s found!"%out_channel))
    return None

def M0(channels,alphas,resonances):
    """Mean of pole positions"""
    return atfi.cast_real(sum(res[0] for res in resonances)/len(resonances))

def M(res_tuple):
    return res_tuple[0]


def get_m(channels,alphas,resonances,a):
    return channels[a][3], channels[a][4]

def q(channels,alphas,resonances,s,a):
    m1,m2 = self.get_m(a)
    # return atfi.cast_complex(two_body_momentum(s,atfi.cast_complex(m1),atfi.cast_complex(m2)))
    s_a = atfi.cast_complex(m1 + m2)
    d_a = atfi.cast_complex(m1-m2)
    return atfi.sqrt(atfi.cast_complex((s-s_a**2) * (s-d_a**2)/(4*s) ))

def get_channel(channels,alphas,resonances,index,L):
    channel_LS = {(c[0],c[1]):i for i,c in enumerate(channels)} # we have to figure out the correct channel for a decay with a given L
    return channel_LS[(index,L)]

def L(channels,alphas,resonances,channel):
    # L is doubled, so for calculations we need L/2
    return channels[channel][1]/2

def BWF(channels,alphas,resonances,s,a,d):
    q = self.q(s,a)
    q0 = self.q(self.M0**2,a)
    return blatt_weisskopf_ff(q,q0,d,atfi.const(L(channels,alphas,resonances,a)))

def gamma(channels,alphas,resonances,s,a):
    return (q(channels,alphas,resonances,s,a))**L(channels,alphas,resonances,a) 

def phaseSpaceFactor(channels,alphas,resonances,s,a):
    return atfi.complex(1/(8* atfi.pi()), atfi.const(0))* q(channels,alphas,resonances,s,a)/atfi.cast_complex(atfi.sqrt(s))

def V(channels,alphas,resonances,s,a,b):
    # R = resonance index
    # a,b = channel indices
    return atfi.cast_complex(sum((res[1][a] * res[1][b])/(res[0]*res[0]-s) for res in resonances))

def Sigma(channels,alphas,resonances,s,a):
    sigma = phaseSpaceFactor(channels,alphas,resonances,s,a) * gamma(channels,alphas,resonances,s,a)**2 
    return atfi.complex(atfi.const(0),atfi.const(1.))*(atfi.cast_complex(sigma))#  + width_factors[a])

def build_D(channels,alphas,resonances,s):
    # we calculate v directly  as 1 - v * Sigma
    # ToDo do this with tf.stack
    # v = atfi.zeros(list(s.shape) + [len(self.channels),len(self.channels)] )
    v = list()
    for a in range(len(channels)):
        temp = list()
        for b in range(len(channels)):
            if a == b:
                # temp.append(atfi.ones(s) * b)
                temp.append(1 -V(channels,alphas,resonances,s,a,b)*Sigma(channels,alphas,resonances,s,b) )
            else:
                # temp.append(atfi.ones(s) * b)
                temp.append(-V(channels,alphas,resonances,s,a,b)*Sigma(channels,alphas,resonances,s,b))
        v.append(atfi.stack(temp,-1))
    v = atfi.stack(v,-2)
    D = atfi.linalg_inv(v)
    return D

def g(self,n,b):
    return self.resonances[n][1][b]

def alpha(self,n):
    return self.alphas[n]

def D(channels,alphas,resonances,s,a,b):            
    return build_D(channels,alphas,resonances,s)[...,a,b]

def P_func(channels,alphas,resonances,s,b):
    p  = channels[b][2] + sum( (res[1][b] * alpha )/atfi.cast_complex(res[0] * res[0]-s)   for res,alpha in zip(resonances,alphas))
    return p
    
@atfi.function
def A_H(self,s,a):
    # s: squared energy
    # a: channel number
    #a_h = self.gamma(s,a) * sum( self.D(s,a,b) * self.P_func(s,b) for b in range(len(self.channels)))
    a_h = sum( self.D(s,a,b) * self.P_func(s,b) for b in range(len(self.channels))) # because The barrier factors are sourced out of the resonance lineshape
    return a_h

@atfi.function
def X(self,s,L):
    # return the Lineshape for the specific outchannel
    channel_number = self.get_channel(self.out_channel,L)
    s = atfi.cast_complex(s)
    return self.A_H(s,channel_number) 