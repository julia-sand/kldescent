import torch

import matplotlib.pyplot as plt

from kldescent.viz.plots import get_all_subplot_handles_labels
from kldescent.viz.contour import *

#update_mpl()

def plot_histogram(plot_loc,hist,xloc,epoch=1):
    
    fig = plt.figure(figsize=(15, 15))
    
    plt.stairs(hist,xloc)
    
    fig.savefig(plot_loc+f"/histogram_{epoch}.png")
    plt.close()

@torch.no_grad
def make_histogram(plot_loc,x,model,boundary):
    
    #plot_loc = self.logger.newpath 
    
    #make a plot of final points vs estimated distribution
    x_samples,_ = model.sample(5000)
    
    qaxtemp = torch.linspace(-5,5,1000,dtype=torch.float32).reshape((1000,1))
    pdf_temp = torch.exp(model.log_prob(qaxtemp))
    pdf_temp[torch.isnan(pdf_temp)] = 0

    fig2 = plt.figure(figsize=(15, 15))
    
    #samples from learned distribution
    plt.hist(x_samples.detach().numpy().flatten(),density=True,bins=100,alpha=0.5,color="orange")
    #plt.hist(x.detach().numpy().flatten(),density=True,bins=100,alpha=0.5)
    
    #pdf of learned distribution
    plt.plot(qaxtemp.detach().numpy().flatten(),pdf_temp.detach().numpy().flatten())    

    fig2.savefig(plot_loc+"/histogram.png")
    plt.close()

@torch.no_grad
def make_scatter_plot(plot_loc,x,model,epoch=1,boundary=None):
    
    #plot_loc = self.logger.newpath 
    
    #make a plot of final points vs estimated distribution
    x_samples,_ = model.sample(1000)

    #qsamples = boundary.generate_samples(1000,mode="initial")
    
    fig2 = plt.figure(figsize=(15, 15))
    
    #samples from distributions
    plt.scatter(x_samples[...,0].detach().numpy(),x_samples[...,1].detach().numpy(),color="blue")
    plt.scatter(x[...,0].detach().numpy(),x[...,1].detach().numpy(),color="orange")
    #plt.scatter(qsamples[...,0],qsamples[...,1])
    #plt.hist(x[...,0].detach().numpy(),density=True,bins=100,alpha=0.5,color="orange")
        
    fig2.savefig(plot_loc+f"/distribution_{epoch}.png")
    plt.close()
    
    #estimate kl divergence between the two sample sets
    #print("KL divergence between the samples",kl_div(x.detach().numpy(),x_samples.detach().numpy()))
    
@torch.no_grad
def distribution_plot(plot_loc,model,Tf,epoch=1,qmax=5):

    #compute marginals 
    xx, yy = torch.meshgrid(torch.linspace(-qmax,qmax, 100), torch.linspace(-qmax,qmax, 100), indexing="xy")
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    
    #find probabilities 
    pdf_temp = torch.exp(model.log_prob(zz))
    pdf_temp[torch.isnan(pdf_temp)] = 0
    #pdf_temp2 = torch.exp(model.log_prob(zz))
    #pdf_temp2[torch.isnan(pdf_temp2)] = 0

    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(25)

    xout,jointout = zz,pdf_temp.detach().clone().numpy().flatten()

    plot_joint_panel(plt.gca(),5,jointout,xout)

    fig.tight_layout()
    fig.savefig(plot_loc+f"/contour_distribution_{epoch}.png",bbox_inches="tight")        

    plt.close()

    fig1 = plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, pdf_temp.view(*xx.shape).detach().clone().numpy(), cmap='coolwarm')
    fig1.savefig(plot_loc+"/pdf_distribution.png",bbox_inches="tight") 
    plt.close()


def format_distribution_ax(self,ax1):
    ax1.set_xlim((-self.qmax,self.qmax))
    ax1.set_xlabel("q")
    ax1.set_ylim(bottom=-0.01)

def fit_kde(self,xs,qmax=5):

    kde = KernelDensity(kernel = "epanechnikov",bandwidth=0.1).fit(xs[np.isfinite(xs)].reshape(-1,1))
    return kde.score_samples(np.linspace(-qmax,qmax,num =1000).reshape(-1,1))

##scatter plot of final distribution 
def marginal_distribution(ax,xs,plottime,qmax=5,benchmark=None):
    """
    Function that plots the an estimate of the marginal densities
    computed by sampled points

    args:
      xs : ndarray
        sampled positions for the position and momentum
      plottime : float
        time at which the samples were taken. Defaults to final time from boundary
    """
    
    #if plottime is None:
    #    plottime = self.boundary.params["Tf"]
    
    ax_pos = ax.inset_axes([0,0,0.48,1],sharey=ax)
    ax_mom = ax.inset_axes([0.52,0,0.48,1],sharey=ax)#divider.append_axes("left", size="50%", pad=0.1, sharey=ax)
    
    #remove ax xaxis
    ax.spines.bottom.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xticks([])

    #add titles
    ax.set_title(f"t = {plottime}")
    ax_pos.set_ylabel("Position Marginal")
    ax_mom.set_ylabel("Momentum Marginal")
    
    #set plot limits
    ax_mom.set_xlim((-qmax,qmax))
    ax_pos.set_xlim((-qmax,qmax))
    
    #add perturbative analytic predictions distributions to plots if available
    if benchmark is not None:
        try:
            xref,posref = benchmark.get_quantity(plottime,"underdamped marginal", data="perturbative underdamped",time_type="t0")   
            ax_pos.plot(xref,posref,color="green",alpha=0.5,label="Benchmark Prediction")
            label2 = "Benchmark Evolved"
        except ValueError: 
            print(f"Benchmark data not available for t = {plottime}")
            label2 = "Free Diffusion"
    else:
        label2 = "Free Diffusion"
    #set iteration properties
    colors = ["midnightblue","orange"]
    labels = ["Controlled by NN.",label2]

    for x in enumerate(xs):
        
        #plot kde fitted distributions
        ax_mom.plot(np.linspace(-qmax,qmax,num=1000), np.exp(fit_kde(x[1][:,1],qmax)), color=colors[x[0]],label=labels[x[0]])
        ax_pos.plot(np.linspace(-qmax,qmax,num=1000), np.exp(fit_kde(x[1][:,0],qmax)), color=colors[x[0]],label=labels[x[0]]) 
        
    #add target boundary distributions to plots
    #ax_pos.fill_between(np.linspace(-qmax,qmax,num=1000),
    #        self.boundary.q_marginal_final(np.linspace(-qmax,qmax,num=1000)),
    #        color="gray",alpha=0.1,label="target")
    ax_mom.fill_between(np.linspace(-qmax,qmax,num=1000),
        normal_distribution(np.linspace(-qmax,qmax,num=1000),mu=0,sigmasq=1),
                    color="gray",alpha=0.1,label="target")
    
    #add formatting
    format_distribution_ax(ax_mom)
    format_distribution_ax(ax_pos)

    #move the yaxis labels to opposite side for mom plot
    ax_mom.yaxis.tick_right()
    ax_mom.yaxis.set_label_position("right")
    
    #set plot limits
    ax_mom.set_xlim((-5,5))
    ax_pos.set_xlim((-3,3))
    
    return 
  
def make_marginal_distribution_fig(xvals,plottime,benchmark=None):

    #make figure object
    fig = plt.figure(figsize=(15,8))
    ax0 = plt.subplot(1,1,1)

    marginal_distribution(ax0,xvals,plottime,benchmark)
    
    h,l = get_all_subplot_handles_labels(fig)

    #make legend 
    legend = fig.legend(h,l,ncols= 4, 
                            frameon=False,handlelength = 1)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))
    
    fig.tight_layout()
    
    return fig  
    
