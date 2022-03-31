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
from amplitf.amplitudes.functional_resonances import kmatrix


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

    D_kma = kmatrix(sp.SPIN_1,-1,5./1000.,alphas,channels,resonances,
                        bls_ds_kmatrix_in,bls_ds_kmatrix_out,out_channel=0)
    
    masses2 = (ma,mc)
    masses1 = (mb,mc)

    # from test_Kmatrix import D_kma # kmatrix with extra channel, that will lead to fall at 2.85GeV
    resonances1 = [ 
                    # BWresonance(sp.SPIN_0,1,atfi.cast_real(2317),30, {(0,1):atfi.complex(atfi.const(-0.017),atfi.const(-0.1256))},{(0,0):atfi.complex(atfi.const(1),atfi.const(0))},*masses1),#D_0(2317) no specific outgoing bls given :(
                    subThresholdBWresonance(sp.SPIN_0,1,atfi.cast_real(2317),30, bls_D2317_in,bls_D2317_out,*masses1,mb,md,d_mesons),
                    # BWresonance(sp.SPIN_2,1,atfi.cast_real(2573),16.9,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1), #D^*_s2(2573)
                    # BWresonance(sp.SPIN_1,-1,atfi.cast_real(2700),122,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1,d_mesons), #D^*_s1(2700)
                    # BWresonance(sp.SPIN_1,-1,atfi.cast_real(2860),159,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1,d_mesons), #D^*_s1(2860)
                    D_kma,
                    BWresonance(sp.SPIN_3,-1,atfi.cast_real(2860),53,bls_D2860_in,bls_D2860_out,*masses1,d_mesons), #D^*_s3(2860)
                    ]  
    resonances2 = [
                    BWresonance(sp.SPIN_HALF,-1,atfi.cast_real(2791.9),8.9,bls_L_2791_in,bls_L_2791_out,*masses2,d_mesons), # xi_c (2790)
                    BWresonance(sp.SPIN_3HALF,-1,atfi.cast_real(2815), 2.43,{},{},*masses2,d_mesons)  # xi_c (2815) no bls couplings given :(
                    ] 
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

        bls_in_1 = [bls_D2317_in,bls_ds_kmatrix_in,bls_D2860_in]
        bls_out_1 = [bls_D2317_out,bls_ds_kmatrix_out,bls_D2860_out]

        bls_in_2 = [bls_L_2791_in]
        bls_out_2 = [bls_L_2791_out]

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
        resonances1[1].alphas = alphas
        resonances1[1].channels = channels
        resonances1[1].resonances = resonances

        # channels[0].update(m11,m12,2,bg1,index=0)
        # channels[1].update(m21,m22,2,bg2,index=1)
        # resonances[0].update(2713.6,[g0,g1])
        # resonances[1].update(2967.1,[g2,g3])
        # channels = [
        # KmatChannel(m11,m12,2,bg1,index=0), # this is the decay channel we will see
        # KmatChannel(m21,m22,2,bg2,index=1) # this is the channel that may cause interference
        # ]
        # resonances = [
        #     KmatPole(2713.6,[g0,g1]),  # D^*_s1(2700)
        #     KmatPole(2967.1,[g2,g3])  # D^*_s1(2860)    # ToDo find if we assigned the g values correctly #D^*_s1(2860)
        # ]
        # resonances1[1].update(sp.SPIN_1,-1,5./1000.,alphas,channels,resonances,
        #              bls_ds_kmatrix_in,bls_ds_kmatrix_out,out_channel=0)

        # resonances1[1].update(sp.SPIN_1,-1,5./1000.,alphas,channels,resonances,
        #         bls_ds_kmatrix_in,bls_ds_kmatrix_out,out_channel=0)
        # D_kma = kmatrix(sp.SPIN_1,-1,5./1000.,alphas,channels,resonances,
        #              bls_ds_kmatrix_in,bls_ds_kmatrix_out,out_channel=0)
        # resonances1[1] = D_kma
        
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
    
    for i in tqdm(range(100)):
        with tf.GradientTape() as t:
            ampl = unbinned_nll(model(args),integral(model(args)))
        
        grad = t.gradient(ampl,args)
        print(any(v is None for v in grad))

        args[-1].assign_add(1.0)

    
    sgma3 = phsp.m2ab(smp) # lmbda_c , D_bar
    sgma2 = phsp.m2ac(smp) # lmbda_c , k
    sgma1 = phsp.m2bc(smp) # D_bar , k
    s1_name = r"$M^2(K^-,\bar{D}^0)$ in GeV$^2$"
    s2_name = r"$M^2(\Lambda_c^+,K^-)$ in GeV$^2$"
    s3_name = r"$M^2(\Lambda_c^+,\bar{D}^0)$ in GeV$^2$"
    print(ampl,max(ampl))
    my_cmap = plt.get_cmap('hot')
    plt.style.use('dark_background')
    plt.xlabel(s1_name)
    plt.ylabel(s3_name)
    plt.scatter(sgma1/1e6,sgma3/1e6,cmap=my_cmap,s=2,c=ampl,marker="s") # c=abs(ampl[mask])
    plt.colorbar()
    plt.savefig("Dalitz.png",dpi=400)
    plt.show()
    plt.close('all')
    for s,name,label in zip([sgma1,sgma2,sgma3],["_D+K","L_c+K","L_c+D"],[s1_name,s2_name,s3_name]):
        n, bins = np.histogram(s**0.5/1e3,weights=ampl,bins=50)
        s = (bins[1:] + bins[:-1])/2.
        plt.plot(s,n,"x")
        plt.xlabel(r""+label.replace("^2","")[:-2])
        plt.savefig("Dalitz_%s.png"%name,dpi = 400)
        plt.show()
        plt.close('all')