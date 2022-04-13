from sympy import S
from amplitf.dynamics import blatt_weisskopf_ff, blatt_weisskopf_ff_squared, breit_wigner_lineshape, relativistic_breit_wigner, mass_dependent_width, orbital_barrier_factor
import amplitf.interface as atfi
from amplitf.kinematics import two_body_momentum, two_body_momentum_squared,two_body_momentum_no_tf
from amplitf.constants import spin as sp, angular as angular_constant
import numpy as np
from amplitf.amplitudes.resonances import BaseResonance
import sympy as sym


@atfi.function
def masses(channels,alphas,resonances,out_channel):
    for channel in channels:
        if channel[0] == out_channel:
            return channel[3],channel[4]
    raise(ValueError("No channel for given index %s found!"%out_channel))
    return None

@atfi.function
def M0(channels,alphas,resonances):
    """Mean of pole positions"""
    return atfi.cast_real(sum(res[0] for res in resonances)/len(resonances))

@atfi.function
def M(res_tuple):
    return res_tuple[0]

@atfi.function
def get_m(channels,alphas,resonances,a):
    return channels[a][3], channels[a][4]

@atfi.function
def q(channels,alphas,resonances,s,a):
    m1,m2 = get_m(channels,alphas,resonances,a)
    # return atfi.cast_complex(two_body_momentum(s,atfi.cast_complex(m1),atfi.cast_complex(m2)))
    s_a = atfi.cast_complex(m1 + m2)
    d_a = atfi.cast_complex(m1-m2)
    return atfi.sqrt(atfi.cast_complex((s-s_a**2) * (s-d_a**2)/(4*s) ))

@atfi.function
def get_channel(channels,alphas,resonances,index,L):
    channel_LS = {(c[0],c[1]):i for i,c in enumerate(channels)} # we have to figure out the correct channel for a decay with a given L
    return channel_LS[(index,L)]

@atfi.function
def L(channels,alphas,resonances,channel):
    # L is doubled, so for calculations we need L/2
    return atfi.cast_complex(atfi.const(channels[channel][1]/2))

@atfi.function
def BWF(channels,alphas,resonances,s,a,d):
    q_ = q(channels,alphas,resonances,s,a)
    q0 = q(channels,alphas,resonances,M0(channels,alphas,resonances)**2,a)
    return blatt_weisskopf_ff(q_,q0,d,atfi.const(L(channels,alphas,resonances,a)))

@atfi.function
def gamma(channels,alphas,resonances,s,a):
    return (q(channels,alphas,resonances,s,a))**L(channels,alphas,resonances,a) 

@atfi.function
def phaseSpaceFactor(channels,alphas,resonances,s,a):
    return atfi.complex(1/(8* atfi.pi()), atfi.const(0))* q(channels,alphas,resonances,s,a)/atfi.cast_complex(atfi.sqrt(s))

@atfi.function
def V(channels,alphas,resonances,s,a,b):
    # R = resonance index
    # a,b = channel indices
    # sm = atfi.complex(atfi.const(0), atfi.const(0))
    # for res in resonances:
    #     sm += (res[1][a] * res[1][b])/(res[0]*res[0]-s)
    # return sm
    return atfi.cast_complex(sum((res[1][a] * res[1][b])/(res[0]*res[0]-s) for res in resonances))

@atfi.function
def V_nonUnitary(channels,alphas,resonances,width_summands,s,a,b):
    # R = resonance index
    # a,b = channel indices
    # sm = atfi.complex(atfi.const(0), atfi.const(0))
    # for res in resonances:
    #     sm += (res[1][a] * res[1][b])/(res[0]*res[0]-s)
    # return sm
    return atfi.cast_complex(sum((res[1][a] * res[1][b])/(res[0]*res[0]-s + width_summands[i]) for i,res in enumerate(resonances)))

@atfi.function
def Sigma(channels,alphas,resonances,s,a):
    sigma = phaseSpaceFactor(channels,alphas,resonances,s,a) * gamma(channels,alphas,resonances,s,a)**2 
    return atfi.complex(atfi.const(0),atfi.const(1.))*(atfi.cast_complex(sigma))#  + width_factors[a])

@atfi.function
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

@atfi.function
def g(self,n,b):
    return self.resonances[n][1][b]

@atfi.function
def alpha(self,n):
    return self.alphas[n]

@atfi.function
def D(channels,alphas,resonances,s,a,b):            
    return build_D(channels,alphas,resonances,s)[...,a,b]

@atfi.function
def P_func(channels,alphas,resonances,s,b):
    p  = channels[b][2] + sum( (res[1][b] * alpha )/atfi.cast_complex(res[0] * res[0]-s)   for res,alpha in zip(resonances,alphas))
    return p

@atfi.function
def P_func_nonUnitary(channels,alphas,resonances,width_summands,s,b):
    p  = channels[b][2] + sum( (res[1][b] * alpha )/atfi.cast_complex(res[0] * res[0]-s + width_summand)   for res,alpha,width_summand in zip(resonances,alphas,width_summands))
    return p
    
@atfi.function
def A_H(channels,alphas,resonances,s,a):
    # s: squared energy
    # a: channel number
    #a_h = self.gamma(s,a) * sum( self.D(s,a,b) * self.P_func(s,b) for b in range(len(self.channels)))
    D_mat = build_D(channels,alphas,resonances,s)

    a_h = sum( D_mat[...,a,b] * P_func(channels,alphas,resonances,s,b) for b in range(len(channels))) # because The barrier factors are sourced out of the resonance lineshape
    return a_h

@atfi.function
def KmatX(channels,alphas,resonances,s,L,out_channel):
    # return the Lineshape for the specific outchannel
    # channel_number = get_channel(channels,alphas,resonances,out_channel,L)
    s = atfi.cast_complex(s)
    return A_H(channels,alphas,resonances,s,out_channel) 

@atfi.function
def SigmanonUnitary(channels,alphas,resonances,width_summands,s,a):
    sigma = phaseSpaceFactor(channels,alphas,resonances,s,a) * gamma(channels,alphas,resonances,s,a)**2  
    return atfi.complex(atfi.const(0),atfi.const(1.))*(atfi.cast_complex(sigma))#  + width_factors[a])

@atfi.function
def build_DnonUnitary(channels,alphas,resonances,width_summands,s):
    # we calculate v directly  as 1 - v * Sigma
    # ToDo do this with tf.stack
    # v = atfi.zeros(list(s.shape) + [len(self.channels),len(self.channels)] )
    v = list()
    for a in range(len(channels)):
        temp = list()
        for b in range(len(channels)):
            if a == b:
                # temp.append(atfi.ones(s) * b)
                temp.append(1 -V(channels,alphas,resonances,s,a,b)*SigmanonUnitary(channels,alphas,resonances,width_summands,s,b) )
            else:
                # temp.append(atfi.ones(s) * b)
                temp.append(-V(channels,alphas,resonances,s,a,b)*SigmanonUnitary(channels,alphas,resonances,width_summands,s,b))
        v.append(atfi.stack(temp,-1))
    v = atfi.stack(v,-2)
    D = atfi.linalg_inv(v)
    return D

@atfi.function
def KmatXnonUnitary(channels,alphas,resonances,width_summands,s,L,out_channel):
    s = atfi.cast_complex(s)
    return A_HnonUnitary(channels,alphas,resonances,width_summands,s,out_channel) 

@atfi.function
def A_HnonUnitary(channels,alphas,resonances,width_summands,s,a):
    # s: squared energy
    # a: channel number
    #a_h = self.gamma(s,a) * sum( self.D(s,a,b) * self.P_func(s,b) for b in range(len(self.channels)))
    D_mat = build_DnonUnitary(channels,alphas,resonances,width_summands,s)

    a_h = sum( D_mat[...,a,b] * P_func_nonUnitary(channels,alphas,resonances,width_summands,s,b) for b in range(len(channels))) # because The barrier factors are sourced out of the resonance lineshape
    return a_h


def Kmatrix_SymPy(channels,alphas,resonances,s,a):
    V = []
    for n,r in enumerate(resonances):
        g_n = sym.symbols(" ".join(["g_{%s%s}"%(n,i) for i in range(len(r[1]))]))
        V.append(g_n)
    V = sym.Matrix(V)
    print(V)
    sigma, V, rho, gamma = sym.symbols('sigma V rho gamma')
