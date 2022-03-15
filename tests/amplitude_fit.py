
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from amplitude_model import three_body_decay_Daliz_plot_function
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
import amplitf.interface as atfi
import matplotlib.pyplot as plt
import numpy as np
from data_reading import read_data_numpy
from iminuit import cost, Minuit
import curses


def run_fit():
    ma = 2286.46 # lambda_c spin = 0.5 parity = 1
    mb = 1864.84 # D^0 bar spin = 0 partiy = -1
    mc = 493.677 # K-  spin = 0 parity = -1
    md = 5619.60  # lambda_b  spin = 0.5 parity = +1

    data = read_data_numpy()
    s1,s2,s3 = data["D0K_M2"],data["LcK_M2"],data["LcD0_M2"]
    md = data["Lb_M"]
    phsp = DalitzPhaseSpace(ma,mb,mc,atfi.convert_to_tensor(md))
    tensor_data = atfi.cast_real(atfi.stack([atfi.convert_to_tensor(s3.values),atfi.convert_to_tensor(s1.values)],axis=1))
    smp = PhaseSpaceSample(phsp,tensor_data)

    maxL, minL = 0,1e15
    global_args = ()

    def print_self(args,L):
        nonlocal maxL, minL,global_args
        
        stdscr.clear()
        stdscr.refresh()
        if -L < minL:
            minL = -L
            global_args = args
        if -L > maxL: 
            maxL = -L
        stdscr.addstr(0,0,"Arguments %s"%(args,))
        stdscr.addstr(1,0,"")
        stdscr.addstr(5,0,"-Log(L)=%.3f, MAX(-Log(L))=%.3f, MIN(-Log(L))=%.3f"%(-L,maxL,minL))
        stdscr.addstr(3,0,"")
        stdscr.addstr(4,0,"")
        stdscr.addstr(2,0,"")
        stdscr.refresh()

    def log_L(*args):
        v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24 = args
        bls_ds_kmatrix_in = {
                        (0,1):atfi.complex(atfi.const(v1),atfi.const(v2)),
                        (2,1):atfi.complex(atfi.const(v3),atfi.const(v4)),
                        (2,3):atfi.complex(atfi.const(v5),atfi.const(v6))
                         }
        bls_ds_kmatrix_out = {
                            (2,0):atfi.complex(atfi.const(v7),atfi.const(v8))
                            }

        bls_D2317_in = {(0,1):atfi.complex(atfi.const(v9),atfi.const(v10))}
        bld_D2317_out = {(0,0):atfi.complex(atfi.const(1),atfi.const(0))}

        bls_L_2791_in = {(0,1):atfi.complex(atfi.const(v11),atfi.const(v12))}
        bls_L_2791_out = {(0,1):atfi.complex(atfi.const(v13),atfi.const(v14))}

        bls_D2860_in = {(4,5):atfi.complex(atfi.const(v15),atfi.const(v16))}
        bls_D2860_out = {(6,0):atfi.complex(atfi.const(v17),atfi.const(v18))}

        kwargs = {"bls_ds_kmatrix_in":bls_ds_kmatrix_in,
                  "bls_ds_kmatrix_out":bls_ds_kmatrix_out, 
                  "bls_D2317_in":bls_D2317_in,
                  "bld_D2317_out":bld_D2317_out,
                  "bls_L_2791_in":bls_L_2791_in,
                  "bls_L_2791_out":bls_L_2791_out,
                  "bls_D2860_in":bls_D2860_in,
                  "bls_D2860_in":bls_D2860_out,
                  "KmatG_factors":(v19,v20,v21,v22) ,
                  "Kmatbg_values":(v23,v24)}
        amplitude = three_body_decay_Daliz_plot_function(smp,phsp,**kwargs)
        L = atfi.nansum(atfi.log(amplitude/atfi.nansum(amplitude)))
        
        print_self(args,L)
        return -L
    
    start = [1 for _ in range(22)] + [0.1,0.1]

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    try:
        m = Minuit(log_L,*start)
        m.migrad()
    finally:
        # automatic parameter names are assigned x0, x1, ...
        curses.echo()
        curses.nocbreak()
        curses.endwin()
    print(m)
    print(global_args)

def amplitude_from_fit_noKmat(args= (7.8129217018561405, -8.9029791987581, 6.672740837966961, 4.753554879115412, 0.25482039945567525, 12.001709375759884, 0.5267874285177503, -0.09320190898238714, 19.44897530068699, 0.05323202382048686, 0.9210743497097295, 0.6849073346905623, 0.8995643811457568, 0.6623408395456626)):
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14 = args
    bls_ds_kmatrix_in = {
                    (0,1):atfi.complex(atfi.const(v1),atfi.const(v2)),
                    (2,1):atfi.complex(atfi.const(v3),atfi.const(v4)),
                    (2,3):atfi.complex(atfi.const(v5),atfi.const(v6))
                    }
    bls_ds_kmatrix_out = {
                        (2,0):atfi.complex(atfi.const(v7),atfi.const(v8))
                        }

    bls_D2317_in = {(0,1):atfi.complex(atfi.const(v9),atfi.const(v10))}
    bld_D2317_out = {(0,0):atfi.complex(atfi.const(1),atfi.const(0))}

    bls_L_2791_in = {(0,1):atfi.complex(atfi.const(v11),atfi.const(v12))}
    bls_L_2791_out = {(0,1):atfi.complex(atfi.const(v13),atfi.const(v14))}

    kwargs = {"bls_ds_kmatrix_in":bls_ds_kmatrix_in,
            "bls_ds_kmatrix_out":bls_ds_kmatrix_out, 
            "bls_D2317_in":bls_D2317_in,
            "bld_D2317_out":bld_D2317_out,
            "bls_L_2791_in":bls_L_2791_in,
            "bls_L_2791_out":bls_L_2791_out }

    ma = 2286.46 # lambda_c spin = 0.5 parity = 1
    mb = 1864.84 # D^0 bar spin = 0 partiy = -1
    mc = 493.677 # K-  spin = 0 parity = -1
    md = 5619.60  # lambda_b  spin = 0.5 parity = +1
    phsp = DalitzPhaseSpace(ma,mb,mc,md) 

    smp = PhaseSpaceSample(phsp,phsp.rectangular_grid_sample(250, 250, space_to_sample="linDP"))

    ampl = three_body_decay_Daliz_plot_function(smp,phsp,**kwargs)
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
    


if __name__=="__main__":
    run_fit()
        


    