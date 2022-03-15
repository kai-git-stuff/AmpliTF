import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data_numpy(filename="/home/kai/LHCb/data/Data_LcD0K_Run2_Dalitz.root"):
    print(filename)
    with uproot.open(filename) as file:
        dir = file['LcD0K']
        tree = dir['DecayTree']
        v = str(tree.values())
        # names = [n for i,n in enumerate(v.split("'")) if i%2 != 0]
        # names = [n for n in names if "M2" in n]
        # print(names)
        data = tree.arrays(["Lb_M", "LcK_M2", "D0K_M2","LcD0_M2"], library="pd")
        mask = (data["Lb_M"] > 5580) & (data["Lb_M"] < 5690)
        data = data[mask]
        return data


if __name__ == "__main__":
    data = read_data_numpy("/home/kai/LHCb/data/Data_LcD0K_Run2_Dalitz.root")
    #plt.hist2d(data["D0K_M2"]**0.5/1e3,data["LcD0_M2"]**0.5/1e3,bins=(50,50))
    plt.hist(data["LcK_M2"]**0.5/1e3,bins=100)
    plt.show()
    print(data)