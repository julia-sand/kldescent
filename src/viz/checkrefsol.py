import matplotlib.pyplot as plt

from .plots import PlotFuns

class RefSol(PlotFuns):

    def __init__(self):
        super(PlotFuns,self).__init__() #fetches the general plotting params from plotfuns
    
    def plot_value(self,times):

      fig,ax = plt.subplots(1,len(times))
    
      for t in range(len(times)):  
        
        xref,vref = self.boundary.get_reference_value(times[t])
        self.plot_reference(ax[t],xref,vref)

        self.format_drift_ax(ax[t],times[t])
      
        ax[t].set_ylabel(r"$\mathbb{U}$")
      
      plt.legend()
      plt.tight_layout()
      plt.savefig(self.subdir+f"/value.png", bbox_inches='tight')
      plt.close()


if __name__=="__main__":
    from ..sde.boundary import UnderdampedNoMomentum

    boundary_cond = UnderdampedNoMomentum()
    RefSol(boundary_cond).plot_value([0,0.5,1])



