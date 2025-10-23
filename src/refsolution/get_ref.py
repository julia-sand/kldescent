import numpy as np 

class RefSolution():
    """
    Class used for fetching the reference solutions for plotting
    """
    def __init__(self,boundary):   
        self.boundary = boundary
        
        #locate the reference solutions
        self.ref_loc = self.get_ref_path()+"/refsol.csv"
        self.cumulants_loc = self.get_ref_path()+"/refcumulants.csv"   
        self.peturbative_loc = self.get_ref_path()+"/peturbative.csv"

    def get_ref_path(self):
        #returns the folder where the reference solution is expected to be

        return os.path.dirname(__file__) + f"{self.boundary.__class__.__name__}"
    
    def get_ref_data(self):
        #returns the folder where the reference solution is expected to be
        arr = np.loadtxt(open(self.ref_path),
                        delimiter=",",skiprows=1) 
        return arr 

    def get_reference_timegrid(self,time=None):
        if time is None:
            time = self.boundary.params["Tf"] + 0.00001
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

    def du_ref(self,q,time):

        #returns the drift of the reference solution 
        try:
            qref,duref = self.get_reference_data(time)
            return np.interp(q,qref,duref,left=duref[0],right=duref[-1])
        except:
            return np.zeros_like(q)

    def get_ref_cumulants(self):
        #returns the folder where the reference solution is expected to be
        arr = np.loadtxt(open(self.cumulants_path),
                        delimiter=",",skiprows=1) 
        return arr 

    def get_known_solution_cumulants(self,cumulant_type):
        arr = self.get_ref_cumulants() 
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

    def load_overdamped(boundary):
        arr = np.loadtxt(boundary.overdamped_path,
                            delimiter=",")  
        return arr 

    def overdamped_drift(q,t,epsilon):
        return 0

    def perturbative_marginal(q,t,epsilon):
        return 0

    def overdamped_pdf(q,t):
        return 0
