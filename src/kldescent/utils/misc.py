import os 

import numpy as np 


def append_array_to_csv(filename,data):
    """append numpy array to an already created csv"""
    
    with open(filename,"a") as file:
        np.savetxt(file,data,fmt="%10.6f",delimiter=',')

def save_array(arr_out,save_location,filename,header):
    """
    Save an array to a CSV file at the specified location.

    The output directory is created automatically if it does not already exist.
    If the file already exists, new values are appended to it.
    
    Parameters
    ----------
    arr_out : np.ndarray
        Array to be saved.
    save_location : str
        Directory path where the file will be written, relative to the
        current working directory.
    filename : str
        Name of the output file (should include the `.csv` extension if desired).
    header : str
        Text to include at the top of the CSV file as a comment.
    """

    #make dir if does not exist
    os.makedirs(save_location,exist_ok=True)
    np.savetxt(save_location+"/"+filename+".csv", arr_out ,fmt="%10.6f",header = header,delimiter=',')
    return

def list_subfolders(folder):
    """Return the full paths of all subfolders in a folder. Only return the top level subfolder."""

    return [f.path for f in os.scandir(folder) if f.is_dir()]