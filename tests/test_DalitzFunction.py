import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

print(sys.path)
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
from amplitf.likelihood import unbinned_nll, integral
from amplitf.kinematics import *
from amplitf.dalitz_decomposition import *
import amplitf.interface as atfi
import matplotlib.pyplot as plt
from amplitf.constants import spin as sp
from amplitf.constants import angular as ang
from matplotlib.colors import LogNorm
import tensorflow as tf
import json
from tqdm import tqdm
from amplitf.amplitudes.dalitz_function import *
from amplitf.amplitudes.resonances import BaseResonance,subThresholdBWresonance,BWresonance
from amplitf.amplitudes.functional_resonances import KmatX, M0


# tf.compat.v1.enable_eager_execution()



# @tf.function(experimental_relax_shapes=True)
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

    ma = 2286.46 # lambda_c spin = 0.5 parity = 1
    mb = 1864.84 # D^0 bar spin = 0 partiy = -1
    mc = 493.677 # K-  spin = 0 parity = -1
    md = 5619.60  # lambda_b  spin = 0.5 parity = +1

    ma,mb,mc,md = phsp.masses

    d_mesons = 1.5/1000.

    decay = dalitz_decay(md,ma,mb,mc,sd,sa,sb,sc,pd,pa,pb,pc,smp,phsp=phsp)
    # we will add the different Amplitudes
    sgma1,sgma2,sgma3 = decay.sgma1,decay.sgma2, decay.sgma3
    bls_ds_kmatrix_in = kwargs['bls_ds_kmatrix_in']
    bls_ds_kmatrix_out = kwargs['bls_ds_kmatrix_out']

    bls_D2317_in = kwargs['bls_D2317_in']
    bls_D2317_out = kwargs['bls_D2317_out']

    bls_L_2791_in = kwargs['bls_L_2791_in']
    bls_L_2791_out = kwargs['bls_L_2791_out']

    bls_D2860_in = kwargs['bls_D2860_in']
    bls_D2860_out = kwargs['bls_D2860_out']

    alphas = kwargs['alphas']
    g0,g1,g2,g3 = kwargs["KmatG_factors"]
    bg1, bg2 = kwargs["Kmatbg_values"]
    m11,m12,m21,m22 = mb,mc,2006.85,mc 

    channels = [(0,2,bg1,m11,m12),
                (1,2,bg2,m21,m22)]
    resonances = [(2713.6,[g0,g1],
                    (2967.1,[g2,g3]))]
    # channels = [
    #     KmatChannel(m11,m12,2,bg1,index=0), # this is the decay channel we will see
    #     KmatChannel(m21,m22,2,bg2,index=1) # this is the channel that may cause interference
    # ]
    # resonances = [
    #     KmatPole(2713.6,[g0,g1]),  # D^*_s1(2700)
    #     KmatPole(2967.1,[g2,g3])  # D^*_s1(2860)    # ToDo find if we assigned the g values correctly #D^*_s1(2860)
    # ]
    m0 = M0(channels,alphas,resonances)
    p0 = two_body_momentum(m0, m11,m12)
    D_kma = (sp.SPIN_1,-1,sp.direction_options(sp.SPIN_1),KmatX(channels,alphas,resonances,sgma1,sp.SPIN_1,0),m0,d_mesons,p0)
    
    masses2 = (ma,mc)
    masses1 = (mb,mc)

    # from test_Kmatrix import D_kma # kmatrix with extra channel, that will lead to fall at 2.85GeV
    resonances1 = [ 
                    # BWresonance(sp.SPIN_0,1,atfi.cast_real(2317),30, {(0,1):atfi.complex(atfi.const(-0.017),atfi.const(-0.1256))},{(0,0):atfi.complex(atfi.const(1),atfi.const(0))},*masses1),#D_0(2317) no specific outgoing bls given :(
                    subThresholdBWresonance(sp.SPIN_0,1,atfi.cast_real(2317),30, bls_D2317_in,bls_D2317_out,*masses1,mb,md,sgma1,d_mesons),
                    # BWresonance(sp.SPIN_2,1,atfi.cast_real(2573),16.9,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1), #D^*_s2(2573)
                    # BWresonance(sp.SPIN_1,-1,atfi.cast_real(2700),122,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1,d_mesons), #D^*_s1(2700)
                    # BWresonance(sp.SPIN_1,-1,atfi.cast_real(2860),159,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1,d_mesons), #D^*_s1(2860)
                    D_kma,
                    BWresonance(sp.SPIN_3,-1,atfi.cast_real(2860),53,bls_D2860_in,bls_D2860_out,*masses1,sgma1,d_mesons), #D^*_s3(2860)
                    ]  
    resonances2 = [
                    BWresonance(sp.SPIN_HALF,-1,atfi.cast_real(2791.9),8.9,bls_L_2791_in,bls_L_2791_out,*masses2,sgma2,d_mesons), # xi_c (2790)
                    BWresonance(sp.SPIN_3HALF,-1,atfi.cast_real(2815), 2.43,{},{},*masses2,sgma2,d_mesons)  # xi_c (2815) no bls couplings given :(
                    ] 
    @atfi.function
    def f(args):
        kwargs = get_kwargs_from_args(args)
        bls_ds_kmatrix_in = kwargs['bls_ds_kmatrix_in']
        bls_ds_kmatrix_out = kwargs['bls_ds_kmatrix_out']
        bls_D2317_in = kwargs['bls_D2317_in']
        bls_D2317_out = kwargs['bls_D2317_out']
        bls_L_2791_in = kwargs['bls_L_2791_in']
        bls_L_2791_out = kwargs['bls_L_2791_out']
        bls_D2860_in = kwargs['bls_D2860_in']
        bls_D2860_out = kwargs['bls_D2860_out']
        alphas = kwargs['alphas']
        g0,g1,g2,g3 = kwargs["KmatG_factors"]
        bg1, bg2 = kwargs["Kmatbg_values"]
        m11,m12,m21,m22 = mb,mc,2006.85,mc
        '''channels = [(index,L,bg,m1,m2)]
        resonances = [(M,couplings)]'''
        channels = [(0,2,bg1,m11,m12),
                    (1,2,bg2,m21,m22)]
        resonances = [(2713.6,[g0,g1]),
                      (2967.1,[g2,g3])]

        resonances1[1]  = (sp.SPIN_1,-1,sp.direction_options(sp.SPIN_1),KmatX(channels,alphas,resonances,sgma1,sp.SPIN_1,0),m0,d_mesons,p0)

        bls_in_1 = [bls_D2317_in,bls_ds_kmatrix_in,bls_D2860_in]
        bls_out_1 = [bls_D2317_out,bls_ds_kmatrix_out,bls_D2860_out]

        bls_in_2 = [bls_L_2791_in]
        bls_out_2 = [bls_L_2791_out]

        def O(nu,lambdas):
            return decay.chain3(smp,nu,*lambdas,[],[],[]) + decay.chain2(smp,nu,*lambdas,resonances2,bls_in_2,bls_out_2) + decay.chain1(smp,nu,*lambdas,resonances1,bls_in_1,bls_out_1)

        ampl = sum(sum(abs(O(ld,[la,0,0]))**2  for la in sp.direction_options(decay.sa))for ld in sp.direction_options(sd))
        return  ampl
    return f

def get_kwargs_from_args(args):
    # tf.print("kwargs %s"%args[0])
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24 = args
    bls_ds_kmatrix_in = {
                    (0,1):atfi.complex(v1,v2),
                    (2,1):atfi.complex(v3,v4),
                    (2,3):atfi.complex(v5,v6)
                    }
    bls_ds_kmatrix_out = {
                        (2,0):atfi.complex(v7,v8)
                        }

    bls_D2317_in = {(0,1):atfi.complex(v9,v10)}
    bld_D2317_out = {(0,0):atfi.complex(atfi.const(1),atfi.const(0))}

    bls_L_2791_in = {(0,1):atfi.complex(v11,v12)}
    bls_L_2791_out = {(0,1):atfi.complex(atfi.const(1),atfi.const(0))}

    bls_D2860_in = {(4,5):atfi.complex(v13,v14)}
    bls_D2860_out = {(6,0):atfi.complex(atfi.const(1),atfi.const(0))}

    kwargs = {"bls_ds_kmatrix_in":bls_ds_kmatrix_in,
            "bls_ds_kmatrix_out":bls_ds_kmatrix_out, 
            "bls_D2317_in":bls_D2317_in,
            "bls_D2317_out":bld_D2317_out,
            "bls_L_2791_in":bls_L_2791_in,
            "bls_L_2791_out":bls_L_2791_out,
            "bls_D2860_in":bls_D2860_in,
            "bls_D2860_out":bls_D2860_out,
            "alphas":[atfi.complex(v15,v16),atfi.complex(v17,v18)],
            "KmatG_factors":[atfi.complex(k,atfi.const(0)) for k in [v19,v20,v21,v22]] ,
            "Kmatbg_values":[atfi.complex(v23,atfi.const(0)),atfi.complex(v24,atfi.const(0))]}
    return kwargs

if __name__ == "__main__":
    ma = 2286.46 # lambda_c spin = 0.5 parity = 1
    mb = 1864.84 # D^0 bar spin = 0 partiy = -1
    mc = 493.677 # K-  spin = 0 parity = -1
    md = 5619.60  # lambda_b  spin = 0.5 parity = +1
    phsp = DalitzPhaseSpace(ma,mb,mc,md) 

    smp = PhaseSpaceSample(phsp,phsp.rectangular_grid_sample(250, 250, space_to_sample="linDP"))

    with open("fit_state.json","r") as f:
        args = json.load(f)
        args = [tf.Variable(v,dtype=atfi.fptype()) for v in args]
    kwargs = get_kwargs_from_args(args)

    model = three_body_decay_Daliz_plot_function(smp,phsp,**kwargs)
    

    @atfi.function
    def nll():
        return unbinned_nll(model(args),integral(model(args)))
    ampl = nll()

    for i in tqdm(range(100)):
        with tf.GradientTape(persistent=False) as t:
            ampl = nll()
        grad = t.gradient(ampl,args)

        for a in args:
            a.assign_add(1.0)