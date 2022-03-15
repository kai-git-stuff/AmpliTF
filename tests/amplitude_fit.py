
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from amplitude_model import three_body_decay_Daliz_plot_function
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
import amplitf.interface as atfi
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


    def print_self(args,L):
        nonlocal maxL, minL
        stdscr.clear()
        if -L < minL: minL = -L
        if -L > maxL: maxL = -L
        stdscr.addstr(0,0,"Arguments %s"%(args,))
        stdscr.addstr(1,0,"")
        stdscr.addstr(5,0,"-Log(L)=%.3f, MAX(-Log(L))=%.3f, MIN(-Log(L))=%.3f"%(-L,maxL,minL))
        stdscr.addstr(3,0,"")
        stdscr.addstr(4,0,"")
        stdscr.addstr(2,0,"")
        stdscr.refresh()

    def log_L(*args):
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
        amplitude = three_body_decay_Daliz_plot_function(smp,phsp,**kwargs)
        L = atfi.nansum(atfi.log(amplitude/atfi.nansum(amplitude)))
        
        print_self(args,L)
        return -L
    
    start = [1 for _ in range(14)]
    
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


if __name__=="__main__":
    run_fit()
        


    