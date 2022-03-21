from amplitf.amplitudes.resonances import BWresonance
from amplitf.fitting.TreeAmplitudes import TreeAmplitude
from amplitf.fitting.DalitzAmplitde import dalitz_decay
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.constants import spin as sp
from operator import itemgetter
import amplitf.interface as atfi
def test_1():
    main_amplitude = TreeAmplitude(True,"main")

    br1 = TreeAmplitude(False,"br1")

    br1.set_dependency("value1")

    def f(amplitude:TreeAmplitude,kwargs):
        print("f1")
        return kwargs["value1"] + kwargs["value2"] * amplitude.branches["br1"](kwargs)

    main_amplitude.set_func(f)
    main_amplitude.add_branch("br1",br1)
    main_amplitude.set_dependency("value2")


    def f2(amplitude:TreeAmplitude,kwargs):
        print("f2")
        return kwargs["value1"]
    br1.set_func(f2)

    args1 = {"value1":1,"value2":2}
    print("---------------------------")
    print(main_amplitude(args1))
    print("---------------------------")

    print(main_amplitude(args1))

    args2 = {"value1":2,"value2":2}
    print("---------------------------")

    print(main_amplitude(args2))

def test_2():
    ma = 2286.46 # lambda_c spin = 0.5 parity = 1
    mb = 1864.84 # D^0 bar spin = 0 partiy = -1
    mc = 493.677 # K-  spin = 0 parity = -1
    md = 5619.60  # lambda_b  spin = 0.5 parity = +1
    phsp = DalitzPhaseSpace(ma,mb,mc,md) 

    smp = PhaseSpaceSample(phsp,phsp.rectangular_grid_sample(10, 10, space_to_sample="linDP"))

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
    d_mesons = 1.5/1000.
    decay = dalitz_decay(md,ma,mb,mc,sd,sa,sb,sc,pd,pa,pb,pc,phsp=phsp)



    rootAmplitude = TreeAmplitude(True,"Main")


    bls_in = TreeAmplitude(False,"Resonance1_bls_in")
    def f1(main_ampl:TreeAmplitude,kwargs):
        S,P = sp.SPIN_HALF,-1
        sgma3 = phsp.m2ab(smp)
        m0,gamma0 = 3500,100
        keys_in = ["bls_in_0_1_r","bls_in_0_1_i"]
        bls_in_0_1_r,bls_in_0_1_i = itemgetter(*keys_in)(kwargs)
        bls_in = {(0,1):atfi.complex(atfi.const(bls_in_0_1_r),atfi.const(bls_in_0_1_i))}
        resonance = BWresonance(S,P,m0,gamma0,bls_in, {},ma,mb,d=5./1000.)
        return resonance.bls_in(md,1.5/1000,md,mc)
    bls_in.set_func(f1)

    bls_out = TreeAmplitude(False,"Resonance1_bls_out")
    def f2(main_ampl:TreeAmplitude,kwargs):
        sgma3 = phsp.m2ab(smp)
        S,P = sp.SPIN_HALF,-1
        m0,gamma0 = 4500,100
        keys_in = ["bls_out_0_1_r","bls_out_0_1_i"]
        bls_out_0_1_r,bls_out_0_1_i = itemgetter(*keys_in)(kwargs)
        bls_out = {(0,1):atfi.complex(atfi.const(bls_out_0_1_r),atfi.const(bls_out_0_1_i))}
        resonance = BWresonance(S,P,m0,gamma0,{}, bls_out,ma,mb,d=5./1000.)
        return resonance.bls_out(sgma3)
    bls_out.set_func(f2)
    bls_in.set_dependency("bls_in_0_1_i")
    bls_in.set_dependency("bls_in_0_1_r")
    bls_out.set_dependency("bls_out_0_1_i")
    bls_out.set_dependency("bls_out_0_1_r")

    Treechain3R1 = decay.chain3(smp,1,1,0,0,bls_in,bls_out,sp.SPIN_HALF,"Resonance1(1,1,0,0)")
    
    print(sum(Treechain3R1({"bls_out_0_1_r":1,"bls_out_0_1_i":0,"bls_in_0_1_r":1,"bls_in_0_1_i":0})))
    print(sum(Treechain3R1({"bls_out_0_1_r":1,"bls_out_0_1_i":0,"bls_in_0_1_r":1,"bls_in_0_1_i":0})))
    print(sum(Treechain3R1({"bls_out_0_1_r":0.5,"bls_out_0_1_i":0,"bls_in_0_1_r":1,"bls_in_0_1_i":0})))
    print(sum(Treechain3R1({"bls_out_0_1_r":1.5,"bls_out_0_1_i":0,"bls_in_0_1_r":1.5,"bls_in_0_1_i":0})))



test_2()

