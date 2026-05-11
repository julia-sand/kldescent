import numpy as np 


def append_array_to_csv(filename,data):
    """append numpy array to a csv"""
    
    with open(filename,"a") as file:
        np.savetxt(file,data,fmt="%10.6f",delimiter=',')
  