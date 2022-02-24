from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
from amplitf.kinematics import *
from amplitf.dalitz_decomposition import *
import amplitf.interface as atfi
import matplotlib.pyplot as plt
from amplitf.constants import spin as sp
from amplitf.constants import angular as ang
from matplotlib.colors import LogNorm
import tensorflow as tf
from amplitf.amplitudes.dalitz_function import *
from amplitf.amplitudes.resonances import *

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

    decay = dalitz_decay(md,ma,mb,mc,sd,sa,sb,sc,pd,pa,pb,pc)
    
    # we will add the different Amplitudes
    bls_ds_kmatrix_in = {
                        (0,1):atfi.complex(atfi.const(-1.8),atfi.const(4.4)),
                        (2,1):atfi.complex(atfi.const(-7.05),atfi.const(-4.06)),
                        (2,3):atfi.complex(atfi.const(4.96),atfi.const(-4.73))
                         }
    bls_ds_kmatrix_out = {
                        (2,0):atfi.complex(atfi.const(-1.064),atfi.const(-0.722))
                        }
    alphas = [atfi.complex(atfi.const(0.00272),atfi.const(-0.00715)), atfi.complex(atfi.const(-0.00111),atfi.const(0.00394))]
    g0,g1,g2,g3 = -8.73, 6.54,6.6,-3.38
    m11,m12,m21,m22 = mb,mc,2006.85,mc 
    channels = [
        KmatChannel(m11,m12,2,0.0135,index=0), # this is the decay channel we will see
        KmatChannel(m21,m22,2,0.0867,index=1) # this is the channel that may cause interference
    ]
    resonances = [
        KmatResonance(2713.6,[g0,g1]),  # D^*_s1(2700)
        KmatResonance(2967.1,[g2,g3])  # D^*_s1(2860)    # ToDo find if we assigned the g values correctly #D^*_s1(2860)
    ]
    D_kma = kmatrix(sp.SPIN_1,-1,alphas,channels,resonances,
                        bls_ds_kmatrix_in,bls_ds_kmatrix_out,out_channel=0)
    
    masses2 = (ma,mc)
    masses1 = (mb,mc)

    from test_Kmatrix import D_kma # kmatrix with extra channel, that will lead to fall at 2.85GeV
    resonances1 = [ 
                    BWresonance(sp.SPIN_0,1,atfi.cast_real(2317),38, {(0,1):atfi.complex(atfi.const(-0.017),atfi.const(-0.1256))},{(0,0):atfi.complex(atfi.const(1),atfi.const(0))},*masses1),#D_0(2317) no specific outgoing bls given :(
                    BWresonance(sp.SPIN_2,1,atfi.cast_real(2573),16.9,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1), #D^*_s2(2573)
                    # BWresonance(sp.SPIN_1,-1,atfi.cast_real(2700),122,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1), #D^*_s1(2700)
                    # BWresonance(sp.SPIN_1,-1,atfi.cast_real(2860),159,bls_ds_kmatrix_in,bls_ds_kmatrix_out,*masses1), #D^*_s1(2860)
                    D_kma,
                    BWresonance(sp.SPIN_3,-1,atfi.cast_real(2860),53,{(4,5):atfi.complex(atfi.const(0.32),atfi.const(-0.33))},
                                                                                {(6,0):atfi.complex(atfi.const(-0.036),atfi.const(0.015))},*masses1), #D^*_s3(2860)
                    ]  
    resonances2 = [
                    BWresonance(sp.SPIN_HALF,-1,atfi.cast_real(2791.9),8.9,{(0,1):atfi.complex(atfi.const(-0.53),atfi.const(0.69))},
                               {(0,1):atfi.complex(atfi.const(-0.0149),atfi.const(-0.0259))},*masses2), # xi_c (2790)
                    BWresonance(sp.SPIN_3HALF,-1,atfi.cast_real(2815), 2.43,{},{},*masses2)  # xi_c (2815) no bls couplings given :(
                    ] 

    ampl = sum(abs(decay.chain3(smp,ld,la,0,0,[]) + decay.chain2(smp,ld,la,0,0,resonances2) + decay.chain1(smp,ld,la,0,0,resonances1))**2
                for la in sp.direction_options(sa) for ld in [1,-1])
    # ampl = sum(
    #             abs(
    #                 sum(
    #                     decay.chain3(smp,ld,la,0,0,[]) + decay.chain2(smp,ld,la,0,0,resonances2) + decay.chain1(smp,ld,la,0,0,resonances1) 
    #                     for ld in sp.direction_options(sd))
    #                 )**2 
    #             for la in sp.direction_options(sa))

    return ampl

ma = 2286.46 # lambda_c spin = 0.5 parity = 1
mb = 1864.84 # D^0 bar spin = 0 partiy = -1
mc = 493.677 # K-  spin = 0 parity = -1
md = 5619.60  # lambda_b  spin = 0.5 parity = +1
phsp = DalitzPhaseSpace(ma,mb,mc,md) 

smp = PhaseSpaceSample(phsp,phsp.rectangular_grid_sample(200, 200, space_to_sample="linDP"))

ampl = three_body_decay_Daliz_plot_function(smp,phsp)
sgma3 = phsp.m2ab(smp) # lmbda_c , D_bar
sgma2 = phsp.m2ac(smp) # lmbda_c , k
sgma1 = phsp.m2bc(smp) # D_bar , k
s1_name = r"$M^2(K^-,\bar{D}^0)$ in GeV$^2$"
s2_name = r"$M^2(\Lambda_c^+,K^-)$ in GeV$^2$"
s3_name = r"$M^2(\Lambda_c^+,\bar{D}^0)$ in GeV$^2$"
print(ampl,max(ampl))
my_cmap = plt.get_cmap('hot')

plt.style.use('dark_background')
plt.xlabel(s2_name)
plt.ylabel(s3_name)
plt.scatter(sgma2/1e6,sgma3/1e6,cmap=my_cmap,s=2,c=ampl,marker="s") # c=abs(ampl[mask])
plt.colorbar()
plt.savefig("Dalitz.png",dpi=400)
plt.show()
plt.close('all')
for s,name,label in zip([sgma1,sgma2,sgma3],["_D+K","L_c+K","L_c+D"],[s1_name,s2_name,s3_name]):
    n, bins = np.histogram(s**0.5/1e3,weights=ampl,bins=100)
    s = (bins[1:] + bins[:-1])/2.
    plt.plot(s,n,"x")
    plt.xlabel(r""+label.replace("^2","")[:-2])
    plt.savefig("Dalitz_%s.png"%name,dpi = 400)
    plt.show()
    plt.close('all')