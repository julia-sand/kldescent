import matplotlib.pyplot as plt
import numpy as np

from .plots import PlotFuns

class RefSol(PlotFuns):

    def __init__(self,boundary):
        #fetches the general plotting params from plotfuns
        super().__init__(boundary) 
    
    def plot_value(self,times):

      fig,ax = plt.subplots(1,len(times))
    
      for t in range(len(times)):  
        
        xref,vref = self.boundary.get_reference_data(times[t])
        self.plot_reference(ax[t],xref,-vref)

        #self.format_drift_ax(ax[t],times[t])
      
        ax[t].set_ylabel(r"$\mathbb{U}$")
        ax[t].set_xlim((-2,2))
        ax[t].set_ylim((-0.2,6))

      ax[-1].plot(np.linspace(-2,2),0.25*((np.linspace(-2,2))**2 -1)**2)
      
      plt.legend()
      plt.tight_layout()
      plt.savefig(self.folder+f"/value.png", bbox_inches='tight')
      plt.close()


if __name__=="__main__":
    from .checkpoint_plots import load_net

    boundary_cond = UnderdampedQuarterGaussian()
    epoch = 4000 
    

    RefSol(boundary_cond).plot_value([0,2.5,5])



