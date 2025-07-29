import yaml 

from ..sde import boundary as boundaries 
from ..utils.parser import *

"""
initialise parameters and boundary conditions from file
"""

def init_params_from_file(folder):
    try:
        # Read YAML file
        with open(f'{folder}/data.yaml', 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        boundary_cond = getattr(boundaries,data_loaded["boundary"])(params=data_loaded)
    except FileNotFoundError: #if no yaml file is provided, use the defaults
        print("No parameter file given. The plots will be created using the default values. See -h for values.")

        data_loaded = None
        
        run_params = overdamped_argparser()
        boundary_cond = getattr(boundaries,run_params["boundary"])()
    
    return boundary_cond, data_loaded

