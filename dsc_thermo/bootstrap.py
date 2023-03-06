import dsc_thermo.heat_flow as hf
from dsc_thermo.molar_mass import molar_mass
from dsc_thermo import thermo
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.constants import R

#convenience function to convert a list of lists into a single list
def flatten(my_list):
    return [l_val for l in my_list for l_val in l]

'''
returns one bootstrap sample of Cp_dict w/ each entry of Cp_dict correspsonding to a cluster
Cp_dict should be formatted as

Cp_dict = {"cluster1":
           [
               [T1, T2, T3 ...],
               [Cp1, Cp2, Cp3 ...]
           ]
          }
where each Ti/Cpi is a list/array corresponding to a single measurement run

'''
def bootstrap_sample(Cp_dict, N_dict=None):
    Cp_sim = {}
    for name, Cp_data in Cp_dict.items():
        N = len(Cp_data[0]) if N_dict is None else N_dict[name]
        indices = np.random.randint(N, size=N)
        T_list = [Cp_data[0][i] for i in indices]
        Cp_list = [Cp_data[1][i] for i in indices]
        Cp_sim.update({name: [T_list, Cp_list]})
    return Cp_sim

#for measurements of glass former in crystal, glass, and undercooled/regular liquid phases
'''
splits "liquid" T_dict/Cp_dict entries into "high"/"low" entries, then bootstrap
simulates the data using the dict entries as clusters
    material: material object to simulate
    T_split: Temperature seperating high-T from undercooled liquid
    N_dict: dict of number of samples to take for each phase 
        (default is to match the number of measurements in the sample)
'''
def sim_glass_former_split_liquid(material, T_split, N_dict=None):
    T_dict = material.T_unflattened
    Cp_dict = material.Cp_unflattened
    
    grouped_Cp_dict = {
        "crystal": [T_dict["crystal"], Cp_dict["crystal"]], 
        "glass": [T_dict["glass"], Cp_dict["glass"]],
        "high": [[],[]],
        "low": [[],[]]
    }
    
    for T, Cp in zip(T_dict["liquid"], Cp_dict["liquid"]):
        if (T > T_split).any():
            key = "high"
        else:
            key = "low"
        grouped_Cp_dict[key][0].append(T)
        grouped_Cp_dict[key][1].append(Cp)
    
    Cp_sim = bootstrap_sample(grouped_Cp_dict, N_dict)
    regrouped_Cp_sim = {
        "crystal": Cp_sim["crystal"], 
        "glass": Cp_sim["glass"],
        "liquid": [Cp_sim["low"][0] + Cp_sim["high"][0], Cp_sim["low"][1] + Cp_sim["high"][1]]
    }
    return regrouped_Cp_sim

#"transposes" a list
def list_transpose(my_list):
    return [[inner_list[i] for inner_list in my_list] for i, _ in enumerate(my_list[0])]

#converts inner lists in a list of lists to arrays
def convert_inner_lists_to_arr(my_list):
    return [np.array(inner_list) if isinstance(inner_list, list) 
                   else inner_list for inner_list in my_list]      

'''
simulates measurements of a material
    Cp_sim_func: function to simulate a set of Cp measurements given the true
        measurements stored in the material object
    value_func: function that calculates the parameter of interest from a
        material object
    Cp_sim_args: additional arguments to Cp_sim_func
    value_args: additional arguments to value_func
    Nsim: number of simulations to perform
    value_shape: shape of the value of interest (if an array)
    
    returns:
        array of simulated values of interest
    or
        array of simulated values of interst + dict of simulated fit values
'''
def simulate(material, Cp_sim_func, value_func, Cp_sim_args=(), value_args=(), 
             Nsim=int(1e4), value_shape=(1,), return_fits=False):
    
    value_arr = np.zeros((Nsim,) + value_shape)
    if return_fits:
        fitval_list = [list(material.Cp_fitvals.values())]*Nsim
        
    for i, _ in enumerate(value_arr):
        #simulate T/Cp
        Cp_sim = Cp_sim_func(material, *Cp_sim_args)
        
        #generate a material with the same properties but simulated T/Cp data
        sim_material_kwargs = material.kwargs
        sim_material_kwargs.update({"Cp_data":Cp_sim})
        sim_material = material.__class__(None, **sim_material_kwargs)
        
        #calculate the relevant value for the simulated material
        value_arr[i] = value_func(sim_material, *value_args)
        
        if return_fits:
            fitval_list[i] = list(sim_material.Cp_fitvals.values())
            
        del sim_material
        
    if return_fits:
        fitval_list = list_transpose(fitval_list)
        fitval_list = convert_inner_lists_to_arr(fitval_list)
        fitval_dict = {key:fitval_list[i] for i, key in enumerate(material.Cp_fitvals.keys())}
        return value_arr, fitval_dict
    else:
        return value_arr

'''
calculate the Mahalanobis distance between a point x and a distribution with
covariance matrix C and mean mu
'''
def mahalanobis_distance(x, C, mu):
    C_inv = np.linalg.inv(C)
    return np.sqrt((x-mu).T@C_inv@(x-mu))
