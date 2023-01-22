import numpy as np
from numpy import sin, cos, pi, exp
from numpy import log as ln
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy import integrate
import warnings
import json
import types
import re
from collections import OrderedDict

from . import molar_mass
from . import heat_flow as hf

R = 8.31452 #J/(K*mol)

#From Glade et al 2000
def cp_crystal(T, a, b):
    return 3*R + a*T + b*T**2

#From Glade et al 2000
def cp_liquid(T, a, b):
    return 3*R + a*T + b*T**(-2)

#stores and calculates thermodynamic quantities for a given phase of a material
class Phase:
    '''
    dsc_data: heat_Flow.Heat_Flow_Data object or list of 
        heat_flow.Heat_Flow_Data objects
    Cp_fit_func: function to fit Cp to
    Cp_data: array containing Cp vs T data (if not None, this will be used 
        instead of calculating Cp from dsc_data)
    
    keyword arguments:
        Tf: Temperature of Fusion in K (lower bound for all integrals) 
            (default 0)
        Hf: Enthalpy of fusion (so H(T)=Hf+ \int_Tf^T Cp dT) (default 0)
        T_min: minimum temperature phase can exist at (default 0)
        T_max: maximum temperature phase can exist at (default np.inf)
        exclude_interval: tuple or list of tuples representing the Temperature 
            interval(s) you want to exclude from the phase.
        p0: initial guess for curve_fit
    '''
    def __init__(self, dsc_data, Cp_fit_func=None, Cp_data=None, molar_mass=None, **kwargs):
        if Cp_data is not None:
            self.T_measured, self.Cp_measured = Cp_data
        elif isinstance(dsc_data, (list, tuple)):
            Cp_list = [data.cp(molar_mass) for data in dsc_data]
            self.T_measured = np.concatenate([data[0] for data in Cp_list])
            self.Cp_measured = np.concatenate([data[1] for data in Cp_list])
        else: 
            self.T_measured, self.Cp_measured = dsc_data.cp(molar_mass)
            
        self.set_params(**kwargs)
        
        T_range = (self.T_measured >= self.T_min) & (self.T_measured <= self.T_max)
        if self.exclude_interval is not None:
            if not isinstance(self.exclude_interval[0], (tuple, list)):
                self.exclude_interval = [self.exclude_interval]
            for interval in self.exclude_interval:
                T_range &= (self.T_measured <= interval[0]) | (self.T_measured >= interval[1])


        self.T_measured = self.T_measured[T_range]
        self.Cp_measured = self.Cp_measured[T_range]
        
        self.Cp_fit_func = Cp_fit_func
        
        self.fit()
        
    def set_params(self, **kwargs):
        self.Tf = kwargs["Tf"] if "Tf" in kwargs else 0
        self.Hf = kwargs["Hf"] if "Hf" in kwargs else 0
        self.T_min = kwargs["T_min"] if "T_min" in kwargs else 0
        self.T_max = kwargs["T_max"] if "T_max" in kwargs else np.inf
        self.exclude_interval = kwargs["exclude_interval"] if "exclude_interval" in kwargs else None
        self.p0 = kwargs["p0"] if "p0" in kwargs else None
        
    def fit(self):
        if self.Cp_fit_func is not None:
            self.Cp = self.get_Cp()
            
            self.enthalpy = self.get_enthalpy()
        
            self.entropy = self.get_entropy()
            
            self.gibbs = lambda T: self.enthalpy(T) - T*self.entropy(T)
        else:
            self.Cp, self.Cp_fitvals, self.enthalpy, self.entropy, self.gibbs = [None]*5
            
    def get_Cp(self):
        #assuming fit_func is in K and T_measured is in C
        self.Cp_fitvals, _ = curve_fit(self.Cp_fit_func, self.T_measured+273.15, self.Cp_measured, p0=self.p0) 
        return lambda T: self.Cp_fit_func(T, *self.Cp_fitvals)
            
    def get_enthalpy(self):
        def H_func(T): 
            if isinstance(T, np.ndarray):
                H = np.zeros(len(T))
                for i, T in enumerate(T):
                    H[i] = self.Hf + integrate.quad(self.Cp, self.Tf, T)[0]
            else:
                H = self.Hf + integrate.quad(self.Cp, self.Tf, T)[0]
            return H
        return H_func
    
    def get_entropy(self):
        def S_func(T):
            Sf = self.Hf/self.Tf if self.Tf != 0 else 0
            Cp_over_T = lambda T: self.Cp(T)/T
            if isinstance(T, np.ndarray):
                S = np.zeros(len(T))
                for i, T in enumerate(T):
                    S[i] = Sf + integrate.quad(Cp_over_T, self.Tf, T)[0]
            else:
                S = Sf + integrate.quad(Cp_over_T, self.Tf, T)[0]
            return S
        return S_func

'''
Essentially a dictionary of Phase objects. If it contains a single Phase objects, the Phase's attributes can be accessed by
        material.attribute
If it contains multiple, they can be accessed as
        material[phase_name].attribute
    or
        material.attribute[phase_name]
Designed to be subclassed for different materials
'''
class Material:
    '''
    dsc_data: hf.heat_flow_data object/list of hf.heat_flow_data objects 
        (single phase) or dict of hf.heat_flow_data objects/dict of lists of 
        hf.heat_flow_data objects (multiple phases) where the keys are phases. 
        If a list is passed, the T/Cp from each heat_flow_data object will be 
        concatenated together.
    molar_mass: molar mass of the material (used to calculate Cp)
    Cp_data: array containing measured Cp vs T data [T, Cp] (single phases) or 
        dict of arrays (multiple phases). If not None, this Cp vs T data will be
        used instead of calculating it from hf.heat_flow_data objects

    keyword arguments:
        see Material.get_phase_params
    '''
    def __init__(self, dsc_data, molar_mass=None, Cp_data=None, **kwargs):
        self.molar_mass = molar_mass
        
        if isinstance(self.Cp_fit_func, dict):
            fit_func_dict = self.Cp_fit_func
        elif isinstance(self.Cp_fit_func, types.FunctionType):
            fit_func_dict = {"": self.Cp_fit_func}
        else:
            raise TypeError("Cp_fit_func must be either dict or static method")
                
        self.phases = {}
        self.phase_names = list(fit_func_dict.keys())
        self.kwargs = kwargs
        
        if not isinstance(dsc_data, dict):
            dsc_data = {phase_name: dsc_data for phase_name in self.phase_names}
        if not isinstance(Cp_data, dict):
            Cp_data = {phase_name: Cp_data for phase_name in self.phase_names}
        for phase_name, Cp_fit_func in fit_func_dict.items():
            phase_params = self.get_phase_params(phase_name, **kwargs)
            self.phases.update({phase_name: Phase(dsc_data[phase_name], 
                Cp_fit_func, Cp_data[phase_name], molar_mass=molar_mass, 
                **phase_params)})

    #Access Phase attributes
    def __getattr__(self, attr):
        if isinstance(self.Cp_fit_func, dict):
            return {name:self.phases[name].__getattribute__(attr) for name in self.phase_names}
        else:
            return list(self.phases.values())[0].__getattribute__(attr)
        
    #behave like a dictionary
    def __getitem__(self, name):
        return self.phases[name]
                 
    #takes keyword arguments passed to __init__ and passes them into phase class.
    def get_phase_params(self, phase_name, **kwargs):
        return {key:kwargs[key][phase_name] if isinstance(kwargs[key], dict) else kwargs[key] for key in kwargs.keys()}
        
    '''
    for single phase:
        function to fit measured Cp to (must be staticmethod)
    for multiple phases: 
        dict. Keys should be phase names and values should be corresponding fit         functions for Cp.
        If a phase should not be fit, then the value should be None
    '''
    Cp_fit_func = None

class Glass_Former(Material):
    '''
    Subclass of Material for Glass Formers. Calculates difference in 
    thermodynamic properties between liquid and solid phases.
    '''
    Cp_fit_func = {"glass":None, "liquid":cp_liquid, "crystal":cp_crystal}
    
    def get_phase_params(self, phase_name, **kwargs):
        phase_params = super().get_phase_params(phase_name, **kwargs)
        if phase_name != "liquid":
            phase_params.update({"Hf":0})
        return phase_params
    
    DeltaCp = lambda self, T: self.Cp["liquid"](T) - self.Cp["crystal"](T)
    DeltaH = lambda self, T: self.enthalpy["liquid"](T) - self.enthalpy["crystal"](T)
    DeltaS = lambda self, T: self.entropy["liquid"](T) - self.entropy["crystal"](T)
    DeltaG = lambda self, T: self.gibbs["liquid"](T) - self.gibbs["crystal"](T)

#Saves the contents of a Material object in .json format
def save(material, filename):
    if os.path.isfile(filename):
        raise RuntimeError("{} already exists".format(filename))
        
    material_dict = {
        "class": str(type(material)),
        "T_measured": {name: list(material.T_measured[name]) for name in material.T_measured},
        "Cp_measured": {name: list(material.Cp_measured[name]) for name in material.Cp_measured},
        "kwargs": material.kwargs
    }
    if "." not in filename:
        filename = filename + ".json"
    with open(filename, "w") as file:
        json.dump(material_dict, file)

'''
creates Material object from data saved in file (see save())
    filename: name of file containing Material object contents
    Material_Class: subclass of Material that the object belongs to
'''
def reload(filename, Material_Class):
    with open(filename, "r") as file:
        material_dict = json.load(file)
        
    Cp_data = {name: np.array([material_dict["T_measured"][name], material_dict["Cp_measured"][name]]) for name in material_dict["T_measured"]}
    kwargs = material_dict["kwargs"]
    return Material_Class(None, Cp_data=Cp_data, **kwargs)

'''
Get dict corresponding to measurement files from a file containing their names. The file 
should be formatted as follows:
'''
#TODO explain how to format the file
#currently doesn't work w/ whitespace in names
def get_files(name_file, path=""):
    file_dict = {}
    measurement_names = ["sample", "sapphire", "empty"]
    position = OrderedDict([("sample",0), ("sapphire",1), ("empty",2)])
    repeat = ['""', "''"]
    with open(name_file, "r") as file:
        file_lines = file.readlines()
    phase_name = ""
    for line in file_lines:
        if line.startswith("#"):
            continue
            
        line.replace(",", " ")
        if "path" in line and re.search("[:=]", line):
            path = re.split("\s*[:=]\s*", line)[-1][:-1] 
            #[:-1] to remove newline
        else:
            split_line = line.split()
            if len(split_line) == 1:
                phase_name = split_line[0]
            #cleaner way to do this?
            elif "sample" in split_line and "sapphire" in split_line and "empty" in split_line:
                measurement_names = split_line
            elif split_line:
                split_line = [path + file_name if file_name not in repeat else last_line[i] for i, file_name in enumerate(split_line)]
                file_names = [split_line[position[measurement_name]] for measurement_name in measurement_names]
                
                if phase_name in file_dict:
                    file_dict[phase_name].append(file_names)
                else:
                    file_dict.update({phase_name:[file_names]})
                
                last_line = split_line
                    
    return file_dict

'''
generates dictionary of hf.Heat_Flow_Data objects from a dictionary containing the names of the corresponding
dsc output files (see get_files)
'''
def gen_heat_flow(file_dict):
    heat_flow_dict = {}
    for phase_name, measurements in file_dict.items():
        heat_flow_dict.update({phase_name:[hf.Heat_Flow_Data(*file_names) for file_names in measurements]})
    return heat_flow_dict

'''
generates a Material (or subclass of Material) object from a file containing the names of the corresponding dsc
output files (see get_files)
    name_file: file containing names of dsc output files
    molar_mass: molar mass of the material
    Material_Class: subclass of Material to initialize from the dsc data
'''
def gen_material(name_file, molar_mass, Material_Class, **kwargs):
    file_dict = get_files(name_file)
    heat_flow_dict = gen_heat_flow(file_dict)
    return Material_Class(heat_flow_dict, molar_mass=molar_mass, **kwargs)
