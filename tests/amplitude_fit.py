
import enum
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from amplitude_model import three_body_decay_Daliz_plot_function
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
import amplitf.interface as atfi
import matplotlib.pyplot as plt
from iminuit import Minuit
import numpy as np
from datetime import datetime
from data_reading import read_data_numpy
import tensorflow as tf
import curses
import json
from tqdm import tqdm
def run_fit():
    ma = 2286.46 # lambda_c spin = 0.5 parity = 1
    mb = 1864.84 # D^0 bar spin = 0 partiy = -1
    mc = 493.677 # K-  spin = 0 parity = -1
    md = 5619.60  # lambda_b  spin = 0.5 parity = +1

    # data = read_data_numpy()
    # s1,s2,s3 = data["D0K_M2"],data["LcK_M2"],data["LcD0_M2"]
    # md_tensor = atfi.convert_to_tensor(data["Lb_M"])
    # phsp = DalitzPhaseSpace(ma,mb,mc,md_tensor)
    # tensor_data = atfi.cast_real(atfi.stack([atfi.convert_to_tensor(s3.values),atfi.convert_to_tensor(s1.values)],axis=1))
    # smp = PhaseSpaceSample(phsp,tensor_data)
    
    phsp = DalitzPhaseSpace(ma,mb,mc,md)
    smp = PhaseSpaceSample(phsp,phsp.rectangular_grid_sample(250, 250, space_to_sample="DP"))

    norm_phsp = DalitzPhaseSpace(ma,mb,mc,md)
    norm_smp = PhaseSpaceSample(norm_phsp,norm_phsp.rectangular_grid_sample(250, 250, space_to_sample="DP"))

    def log_L(args):
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
                    "KmatG_factors":[atfi.cast_complex(k) for k in [v19,v20,v21,v22]] ,
                    "Kmatbg_values":[atfi.cast_complex(v23),atfi.cast_complex(v24)]}
            return kwargs
        kwargs = get_kwargs_from_args(args)
        amplitude= three_body_decay_Daliz_plot_function(smp.data,phsp,**kwargs)
        norm_Amplitude = three_body_decay_Daliz_plot_function(norm_smp.data,norm_phsp,**kwargs)
        def f(args):
            kwargs = get_kwargs_from_args(args)
            L = atfi.nansum(atfi.log(amplitude(kwargs)) - atfi.log(atfi.reduce_sum(norm_Amplitude(kwargs))))
            return -L
        return f

    start = [-1.8,4.4,-7.05,-4.06,4.96,-4.73,-1.064,-0.722,-0.017,-0.1256,-0.53,0.69,-0.0149,-0.0259,0.32,-0.33,-0.00111,0.00394,-8.73, 6.54,6.6,-3.38,0.0135,0.0867]
    start = (-4.0997390647818905, -8.569145833770673, -7.050187572578236, -4.060037351162219, 4.9597607808060955, -4.730072126808973, -0.21160013324459656, -0.86648314691716, 
    -7.180039890338165, 22.996448656493513, -0.08747872885883322, -0.566200303177854, -0.37518335023873656, 0.13094320460476175, -751.5674862877551, -1970.1003751657167, -24200.672340184457, 
    -3181.350661796871, -8.051330172840395, -14.09372484324325, 15.281755542857619, -43.79566512218444, 0.11042225014323345, -0.5304791850986086)
    vars = [atfi.Variable(v,dtype=atfi.fptype()) for v in start]
    gradient = None
    f = log_L(vars)

    for i in tqdm(range(100)):
        with tf.GradientTape(persistent=False) as tape:
            L = f(vars)
        grads = tape.gradient(L, vars)
        gradient = [g.numpy() for g in grads]
        print(L)
        vars[-1].assign_add(0.1)



    class fit_func:
        def __init__(self) -> None:
            self.f = log_L(vars)
        def __call__(self,*args):
            nonlocal vars, gradient
            tmp = [v.assign(new_v) for v,new_v in zip(vars,args)]
            with tf.GradientTape(persistent=True) as tape:
                L = self.f(vars)
            grads = tape.gradient(L, vars)
            gradient = [g.numpy() for g in grads]
            print(L)
            return L
        
        def grad(self,*args):
            nonlocal vars, gradient
            tmp = [v.assign(new_v) for v,new_v in zip(vars,args)]
            with tf.GradientTape(persistent=True) as tape:
                L = self.f(vars)
            grads = tape.gradient(L, vars)
            gradient = [g.numpy() for g in grads]
            print(gradient)
            print("Gradient for L= %s"%L)

            gradient = [g.numpy() for g in grads]
            return gradient
    f = fit_func()
    m = Minuit(f,*vars,grad=f.grad)
    m.strategy = 0
    m.migrad()

    # amplitude_from_fit_Kmat(global_args)

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
    bls_D2317_out = {(0,0):atfi.complex(atfi.const(1),atfi.const(0))}

    bls_L_2791_in = {(0,1):atfi.complex(atfi.const(v11),atfi.const(v12))}
    bls_L_2791_out = {(0,1):atfi.complex(atfi.const(v13),atfi.const(v14))}

    kwargs = {"bls_ds_kmatrix_in":bls_ds_kmatrix_in,
            "bls_ds_kmatrix_out":bls_ds_kmatrix_out, 
            "bls_D2317_in":bls_D2317_in,
            "bls_D2317_out":bls_D2317_out,
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
    
def amplitude_from_fit_Kmat(args= (-4.099734677404456, -8.56913786985447, -7.050191901660969, -4.060037916093625, 4.9597493852321675, -4.730072439124291, -0.20512145609598334, 
-0.8702034947509745, -6.966083252607992, 22.899564644471777, -0.08485932487335121, -0.5507987242904665, -0.36955650058538664, 0.13408074003937026, -548.9523910379469, -2276.4262004965685,
-24111.325434992275, -3698.3453267615373, -9.203941784379237, -12.31759697812728, 15.233808345924345, -44.94879841468681, 0.10964321155182125, -0.52524516351121)):
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
                "KmatG_factors":(atfi.cast_complex(k) for k in [v19,v20,v21,v22]) ,
                "Kmatbg_values":(v23,v24)}

    ma = 2286.46 # lambda_c spin = 0.5 parity = 1
    mb = 1864.84 # D^0 bar spin = 0 partiy = -1
    mc = 493.677 # K-  spin = 0 parity = -1
    md = 5619.60  # lambda_b  spin = 0.5 parity = +1
    phsp = DalitzPhaseSpace(ma,mb,mc,md) 

    smp = PhaseSpaceSample(phsp,phsp.rectangular_grid_sample(250, 250, space_to_sample="linDP"))

    ampl = three_body_decay_Daliz_plot_function(smp.data,phsp,**kwargs)
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
        


    