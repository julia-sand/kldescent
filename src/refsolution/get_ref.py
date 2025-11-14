import os

import numpy as np 

from ..sde.boundary import *
from .marginal_overdamped.gauss import Gaussian_Overdamped_Ref
from .utilfunctions import *

class GaussRefSolution():
    """
    Class used for fetching the reference solutions for plotting
    """

    def __init__(self,boundary=None,folder_name=None):   
        
        if folder_name is not None: #use this if running the file directly
            self.folder_name = folder_name
        #self.folder_name = f"{self.boundary.__class__.__name__}"
        
        self.boundary= boundary
        #only for gaussian
        self.refsol_phi = Gaussian_Overdamped_Ref(self.boundary)

        #locate the reference solutions
        self.ref_loc = self.get_ref_path()+"/refsol.csv"
        self.cumulants_loc = self.get_ref_path()+"/refcumulants.csv"   
        self.overdamped_loc = self.get_ref_path()+"/overdamped.csv"

    def get_ref_path(self):
        #returns the folder where the reference solution is expected to be

        return os.path.dirname(__file__) + "/" + self.folder_name
    
    def get_ref_data(self,data=None):
        #returns the folder where the reference solution is expected to be
        if data == "overdamped":
            arr = np.loadtxt(open(self.overdamped_loc),
                        delimiter=",",skiprows=1) 
        else: 
            arr = np.loadtxt(open(self.ref_loc),
                        delimiter=",",skiprows=1) 
        return arr 

    def get_reference_timegrid(self,time=None):
        try: 
            arr = self.get_ref_data()
            return np.unique(arr[:,0])
        except FileNotFoundError:
            return np.linspace(0,time)
    
    def get_reference_du(self,time):

        #time is the time you want the data

        ##fetch the reference data for halfbridge
        arr = self.get_ref_data()  
       
        ##fetch the reference data approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time,8))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x

        #RETURNS X, DU
        return np.squeeze(arrtemp[:,1]),np.squeeze(arrtemp[:,-1])

    def get_overdamped_quantity(self,time,quantity):

        #time is the time you want the data

        ##fetch the reference data for halfbridge
        arr = self.get_ref_data(data="overdamped")  
       
        ##fetch the reference data approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time,8))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x

        if quantity =="value":
            idx=-1
        elif quantity =="marginal":
            idx=-2
        
        #RETURNS X, DU
        return np.squeeze(arrtemp[:,1]),np.squeeze(arrtemp[:,idx])

    def du_ref(self,q,time):

        #returns the drift of the reference solution 
        try:
            qref,duref = self.get_reference_data(time)
            return np.interp(q,qref,duref,left=duref[0],right=duref[-1])
        except:
            return np.zeros_like(q)

    def get_ref_cumulants_file(self):
        #returns the folder where the reference solution is expected to be
        arr = np.loadtxt(open(self.cumulants_path),
                        delimiter=",",skiprows=1) 
        return arr 

    def get_known_solution_cumulants(self,cumulant_type):
        arr = self.get_ref_cumulants_file() 
        try:  
            if cumulant_type == "Position Variance": 
                temp = np.squeeze(arr[:,1])#np.squeeze(arr[:,4])
            elif cumulant_type == "Momentum Variance":
                temp = np.squeeze(arr[:,3])#np.squeeze(arr[:,6])
            elif cumulant_type == "Cross Correlation":
                temp = np.squeeze(arr[:,2])#np.squeeze(arr[:,5])
            elif cumulant_type == "Momentum Mean":
                temp = np.squeeze(arr[:,8])
            elif cumulant_type == "Position Mean":
                temp = np.squeeze(arr[:,7])
            else:
                temp = np.zeros_like(np.squeeze(arr[:,0]))
        except IndexError:
            return np.zeros_like(np.squeeze(arr[:,0])),np.zeros_like(np.squeeze(arr[:,0]))

        return np.squeeze(arr[:,0]),temp

    def get_reference_data(self,time):
        arr = self.get_ref_data()
        
        return np.linspace(-4,4),-(np.squeeze(arr[np.where(arr[:,0]==time),-1]) + np.squeeze(arr[np.where(arr[:,0]==time),-2])*np.linspace(-4,4))

    def load_overdamped(self,folder):
        arr = np.loadtxt(folder+"/overdamped.csv",
                            delimiter=",")  
        return arr 

    def perturbative_marginal(q,t,epsilon):
        return 0



def plot_overdamped(boundary,folder_name, quantity):
    reference_ = GaussRefSolution(boundary,folder_name=folder_name)

    times = [0,0.06,0.12,0.2]

    fig,axs = plt.subplots(ncols=len(times),nrows=1)
    fig.set_figheight(15)
    fig.set_figwidth(25)

    for time in enumerate(times):
        q, ptq = reference_.get_overdamped_quantity(time[1], quantity)
        axs[time[0]].plot(q,ptq)
        axs[time[0]].set_title(f"t={time[1]}")
        if quantity=="marginal":
            axs[time[0]].plot(q,normalise(q,reference_.refsol_phi.phi_t(q,time[1])*reference_.refsol_phi.hatphi_t(q,time[1])))
            #plot(q,np.exp(-q**2/2)/(np.sqrt(2*np.pi)))
            #axs[time[0]].plot(q,np.exp(-q**2/(2*0.25**2))/(np.sqrt(2*np.pi*0.25**2)))
        if quantity=="value":
            axs[time[0]].plot(q,reference_.refsol_phi.value_fun(q,time[1]))
            #plot(q,np.exp(-q**2/2)/(np.sqrt(2*np.pi)))
            #axs[time[0]].plot(q,np.exp(-q**2/(2*0.25**2))/(np.sqrt(2*np.pi*0.25**2)))

    fig.tight_layout()

    fig.savefig(reference_.get_ref_path()+f"/overdamped_{quantity}.png",bbox_inches="tight")        

if __name__=="__main__":
    import matplotlib.pyplot as plt
    boundary = UnderdampedGaussian()
    for quantity in ["value","marginal"]:
        plot_overdamped(boundary = boundary,folder_name="UnderdampedGaussian",quantity=quantity)
