# dsc_thermo

This is a package for processing isothermal heat capacity measurements from a Perkin-Elmer DSC 8000 and analyzing the results. The primary goals are wide-applicability and ease of use.

## Installation
`dsc_thermo` can be installed using `pip`
 - `pip install git@github.com:colbystoddard/dsc_thermo.git`

## Package Contents
 - `heat_flow.py` calculates isothermal heat capacity vs. temperature from the data files generated by Perkin-Elmer DSC 8000 measurements.
 - `thermo.py` calculates quantities of interest, such as entropy, enthalpy, and gibbs free energy from heat capacity data.
 - `bootstrap.py` implements bootstrap methods for estimating measurement uncertainties.
 - `molar_mass.py` calculates the molar mass of molecules/alloys
