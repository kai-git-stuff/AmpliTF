import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data_numpy(filename="LcD0K15.root",folder="/home/kai/LHCb/data/Marian/InUse/"):
    filename= folder+filename
    print(filename)

    with uproot.open(filename) as file:
        tree = file['t']
        v = str(tree.values())
        names = [n for i,n in enumerate(v.split("'")) if i%2 != 0]
        names = [n for n in names if "Lc_M" in n]
        print(names)
        wanted_params = ["Lb_M","D0_MM"]
        def add_partice(v):
            nonlocal wanted_params
            wanted_params.extend(["%s_%s"%(v,p) for p in ["PE","PX","PY","PZ"]])
        add_partice("D0")
        add_partice("K")
        add_partice("Lc")

        data = tree.arrays(wanted_params, library="pd")
        mask = (data["Lb_M"] > 5580) & (data["Lb_M"] < 5690) & (data["D0_MM"] > 1840) & (data["D0_MM"] < 1900)
        data = data[mask]

        def mass(v1,v2):
            return ( (data["%s_PE"%v1] + data["%s_PE"%v2])**2 - 
                    (data["%s_PX"%v1] + data["%s_PX"%v2])**2  - 
                    (data["%s_PY"%v1] + data["%s_PY"%v2])**2 - 
                    (data["%s_PZ"%v1] + data["%s_PZ"%v2])**2)

        def single_mass(v):
            return data["%s_PE"%v]**2 - data["%s_PX"%v]**2 - data["%s_PY"%v]**2 - data["%s_PZ"%v]**2
        D0_M2 = single_mass("D0")
        K_M2 = single_mass("K")
        Lc_M2 = single_mass("Lc")
        s1,s2,s3 = mass("D0","K"), mass("Lc","K"), mass("Lc","D0")

        return s1,s2,s3


if __name__ == "__main__":
    s1,s2,s3 = data =read_data_numpy()
    #plt.hist2d(s1**0.5/1e3,s3**0.5/1e3,bins=(100,100))
    plt.hist(s1**0.5/1e3,bins=100)
    plt.show()
    print(data)