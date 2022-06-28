import re
import pkgutil

def gen_atomic_mass_dict():
    atomic_masses = {}
    element_info = pkgutil.get_data(__package__, "data/element_info.txt").decode("utf-8")
    element_data_list = element_info.split("\r\n\r\n")
    for element_data in element_data_list:
        if 'Atomic Symbol' in element_data:
            split_data = element_data.splitlines()
            element_symbol = split_data[1].split(" = ")[-1]
            mass_range = re.sub("[()\[\]]", "", split_data[5].split(" = ")[-1]).split(",")
            if len(mass_range) == 2:
                atomic_mass = (float(mass_range[0]) + float(mass_range[1]))/2
            else:
                try:
                    atomic_mass = float(mass_range[0])
                except ValueError:
                    atomic_mass = None
                
            atomic_masses.update({element_symbol: atomic_mass})
    return atomic_masses

atomic_masses = gen_atomic_mass_dict()


def molar_mass(formula):
    formula = "".join(formula.split("_")) #remove underscores
    
    if re.sub("[A-Za-z]+|[0-9]+|[.]", "", formula) != '':
        raise ValueError('Formula Contains Invalid Characters')
        
    elements = re.findall("|".join(sorted(atomic_masses.keys(), key=len)[::-1]), formula)
    invalid_elements = re.findall("[A-Za-z]+", re.sub("|".join(sorted(elements, key=len)[::-1]), '', formula))
    if invalid_elements:
        raise ValueError("Unknown Elements {}".format(invalid_elements))

    
    mass = 0
    for element in elements:
        if element != '':
            start = re.search(element, formula).end()
            subscript_match = re.match("([0-9]*[.])?[0-9]*", formula[start:])
            subscript = subscript_match.group() if (subscript_match is not None and subscript_match.group() != '') else 1
            try:
                mass += atomic_masses[element]*float(subscript)
            except ValueError:
                raise ValueError("Invalid subscript after {}: {}".format(element, subscript))
            except TypeError:
                raise ValueError("Atomic mass of {} unknown".format(element)) 

    if "." in formula:
        mass /= 100   #assuming decimals means the subscripts represent percentages
    return mass

if __name__ == "__main__":
    print(molar_mass("Zr52.5Cu17.9Ni14.6Al10Ti5"))
