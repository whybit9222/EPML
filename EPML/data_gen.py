import pandas as pd
import numpy as np
import cantera as ct
import os 

class DataGen:
    
    def __init__(self):
        pass
    
    mechanism_path = "./mechanisms"
    data_path = "./data/raw"
    mechanism = ""
    gas = 0

    def select_mechanism(self, mech_name):
        if mech_name in os.listdir(self.mechanism_path):
            self.mechanism = mech_name
            print(self.mechanism_path + "/" + self.mechanism)
            self.gas = ct.Solution(self.mechanism_path + "/" + self.mechanism)
            print(self.gas)
        else:
            print("Not found")

    def single_prediction(self, T, P, HOratio, IOratio):
        self.gas.TP = T, P
        self.gas.set_equivalence_ratio(
            phi = HOratio / 2.0,
            fuel = "H2",
            oxidizer = {'O2':1.0, "N2":IOratio}
        )
        r = ct.IdealGasConstPressureReactor(contents=self.gas, name="Batch Reactor")
        reac_net = ct.ReactorNet([r])
        var_name_state = [r.component_name(item) for item in range(r.n_vars)]
        time_history = pd.DataFrame(columns=var_name_state)
        t = np.zeros(1)
        j = 0
        while t[j] < 1:
            j = j+1
            t = np.append(t, reac_net.step())
            time_history.loc[t[j]] = r.get_state()

        tau = time_history["H"].idxmax()

        if tau <= 1:
            return True
        else:
            return False

    def calc_T(self, P, HOratio, IOratio, Tmin=100, Tmax=1000, Tprec=1):
        deltaT = Tmax - Tmin
        while(deltaT > Tprec):
            Ttemp = 0.5*(Tmax + Tmin)
            explosion = self.single_prediction(Ttemp, P, HOratio, IOratio)
            if explosion:
                Tmax = Ttemp
            else:
                Tmin = Ttemp
            deltaT = Tmax - Tmin
        return (Tmax + Tmin) * 0.5

    def gen_classification_data(self, cases, filename):
        #format: [T, P, HOr, IOr]
        output = np.zeros((len(cases),5))
        for idx, case in enumerate(cases):
            output[idx][1] = case[0]
            output[idx][2] = case[1]
            output[idx][3] = case[2]
            output[idx][4] = case[3]
            output[idx][0] = self.single_prediction(case[0], case[1], case[2], case[3])
        
        if not filename+".npy" in os.listdir(self.data_path):
            np.save(self.data_path + "/" + filename, output)
        else:
            np.save(self.data_path + "/" + filename + "_new", output)
        
        return output
    
